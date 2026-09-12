from __future__ import annotations

import base64
import json
import os
import re
from typing import Protocol
import urllib.request

import cv2
import numpy as np

from mapping.config import OCRConfig
from mapping.schema import PlayerUnit, RawPlayer
from mapping.player_detection import crop_box


class OCRBackend(Protocol):
    def read_text(self, image: np.ndarray, *, digits_only: bool = False) -> tuple[str, float]: ...
    def read_player(self, unit: np.ndarray, number: np.ndarray, name: np.ndarray) -> RawPlayer: ...


def _payload(result: object) -> dict:
    raw = result.json
    raw = raw if isinstance(raw, dict) else raw()
    return raw.get("res", raw)


def _number(value: str) -> int | None:
    compact = re.sub(r"[^A-Z0-9|]", "", value.upper())
    compact = compact.translate(str.maketrans({"I": "1", "L": "1", "O": "0", "|": "1"}))
    match = re.search(r"(?<!\d)(\d{1,2})(?!\d)", compact)
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 99 else None


def _upscale(image: np.ndarray, factor: float) -> np.ndarray:
    if image.size == 0 or factor <= 1:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


class PaddleOCRBackend:
    """PP-OCRv6 Small detector and recognizer; no server/large model is loaded."""

    def __init__(self, config: OCRConfig) -> None:
        from paddleocr import PaddleOCR
        self.config = config
        self.model = PaddleOCR(
            text_detection_model_name=config.detection_model,
            text_recognition_model_name=config.recognition_model,
            device=config.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    @staticmethod
    def _parse(data: dict, digits_only: bool) -> tuple[str, float]:
        texts = [str(value).strip() for value in data.get("rec_texts", [])]
        scores = [float(value) for value in data.get("rec_scores", [])]
        if digits_only:
            candidates = [(number, score) for text, score in zip(texts, scores)
                          if (number := _number(text)) is not None]
            if not candidates:
                return "", 0.0
            winner = max(candidates, key=lambda item: item[1])
            return str(winner[0]), winner[1]
        return " ".join(text for text in texts if text), (sum(scores) / len(scores) if scores else 0.0)

    def read_text_batch(
        self, images: list[np.ndarray], *, digits_only: bool = False,
    ) -> list[tuple[str, float]]:
        if not images:
            return []
        prepared = [_upscale(image, self.config.upscale_factor) for image in images]
        results = list(self.model.predict(prepared))
        if len(results) != len(images):
            raise RuntimeError("PaddleOCR result count did not match crop count")
        return [self._parse(_payload(result), digits_only) for result in results]

    def read_text(self, image: np.ndarray, *, digits_only: bool = False) -> tuple[str, float]:
        if image.size == 0:
            return "", 0.0
        return self.read_text_batch([image], digits_only=digits_only)[0]

    def read_player(self, unit: np.ndarray, number: np.ndarray, name: np.ndarray) -> RawPlayer:
        number_text, number_confidence = self.read_text(number, digits_only=True)
        player_name, name_confidence = self.read_text(name)
        return RawPlayer(_number(number_text), player_name.strip(), number_confidence, name_confidence)

    def read_player_batch(
        self, crops: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[RawPlayer]:
        numbers = self.read_text_batch([number for _, number, _ in crops], digits_only=True)
        names = self.read_text_batch([name for _, _, name in crops])
        return [RawPlayer(_number(number_text), player_name.strip(), number_confidence, name_confidence)
                for (number_text, number_confidence), (player_name, name_confidence)
                in zip(numbers, names)]


class LLMVisionOCRBackend:
    """OpenAI-compatible vision hook returning structured JSON for difficult graphics."""

    def __init__(self, config: OCRConfig) -> None:
        if not config.llm_api_url or not config.llm_model:
            raise ValueError("llm_vision requires ocr.llm_api_url and ocr.llm_model")
        self.config = config

    def _request(self, image: np.ndarray, prompt: str) -> dict:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode OCR crop")
        image_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        body = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.llm_api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.config.llm_api_url, json.dumps(body).encode("utf-8"), headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", payload)
        return json.loads(content) if isinstance(content, str) else content

    def read_text(self, image: np.ndarray, *, digits_only: bool = False) -> tuple[str, float]:
        field = "number containing only 1-2 digits" if digits_only else "visible text"
        result = self._request(image, f'Return JSON {{"text": "...", "confidence": 0.0}} for the {field}.')
        return str(result.get("text", "")), float(result.get("confidence", 0.0))

    def read_player(self, unit: np.ndarray, number: np.ndarray, name: np.ndarray) -> RawPlayer:
        result = self._request(
            unit,
            'Read this football lineup player card. Return JSON only: '
            '{"jersey_number": integer_or_null, "player_name": "...", '
            '"number_confidence": 0.0, "name_confidence": 0.0}.',
        )
        value = result.get("jersey_number")
        jersey = int(value) if value is not None and str(value).isdigit() else None
        return RawPlayer(
            jersey, str(result.get("player_name", "")).strip(),
            float(result.get("number_confidence", 0.0)), float(result.get("name_confidence", 0.0)),
        )


def create_ocr_backend(config: OCRConfig) -> OCRBackend:
    return PaddleOCRBackend(config) if config.backend == "paddleocr" else LLMVisionOCRBackend(config)


def ocr_player_unit(image: np.ndarray, unit: PlayerUnit, backend: OCRBackend) -> RawPlayer:
    return backend.read_player(
        crop_box(image, unit.box), crop_box(image, unit.number_box), crop_box(image, unit.name_box),
    )


def ocr_player_units(image: np.ndarray, units: list[PlayerUnit], backend: OCRBackend) -> list[RawPlayer]:
    crops = [(crop_box(image, unit.box), crop_box(image, unit.number_box), crop_box(image, unit.name_box))
             for unit in units]
    batch_method = getattr(backend, "read_player_batch", None)
    if callable(batch_method):
        return batch_method(crops)
    return [backend.read_player(*items) for items in crops]
