import argparse
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional, cast

try:
    from google import genai  # New SDK: google-genai
except Exception:  # pragma: no cover
    genai = None

try:
    import google.generativeai as legacy_genai  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    legacy_genai = None


# Model configuration copied from the notebook.
MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety settings copied from the notebook.
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]


DEFAULT_SYSTEM_PROMPT = """
You are a specialist in comprehending receipts.
Input images in the form of receipts will be provided to you,
and your task is to respond to questions based on the content of the input image.
""".strip()


DEFAULT_BALANCE_PROMPT = "What is the balance amount in the image?"
DEFAULT_JSON_PROMPT = "Convert Invoice data into json format with appropriate json tags as required for the data in image"


def using_new_sdk() -> bool:
    return bool(genai is not None and hasattr(genai, "Client"))


def using_legacy_sdk() -> bool:
    return bool(legacy_genai is not None and hasattr(legacy_genai, "configure"))


def load_api_key() -> str:
    """Load API key from environment variables or local .env file."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key.strip().strip('"').strip("'")

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
                return value.strip().strip('"').strip("'")

    raise RuntimeError(
        "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or .env."
    )


def configure_gemini() -> Optional[Any]:
    api_key = load_api_key()

    if using_new_sdk():
        assert genai is not None
        return genai.Client(api_key=api_key)

    if using_legacy_sdk():
        assert legacy_genai is not None
        legacy_genai.configure(api_key=api_key)
        return None

    raise RuntimeError(
        "No supported Gemini SDK found. Install either `google-genai` or `google-generativeai`."
    )


def list_supported_models(client: Optional[Any]) -> None:
    if using_new_sdk() and client is not None:
        for model_info in client.models.list():
            print(model_info.name)
        return

    if using_legacy_sdk():
        assert legacy_genai is not None
        for model_info in legacy_genai.list_models():
            if "generateContent" in model_info.supported_generation_methods:
                print(model_info.name)
        return

    raise RuntimeError("Unable to list models because no supported Gemini SDK is available.")


def image_format(image_path: str):
    """Build Gemini image part payload from a local image file."""
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    mime_type, _ = mimetypes.guess_type(str(img))
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        mime_type = "image/png"

    image_parts = [
        {
            "mime_type": mime_type,
            "data": img.read_bytes(),
        }
    ]
    return image_parts


def build_model(client: Optional[Any], model_name: str):
    if using_new_sdk() and client is not None:
        return {"client": client, "model_name": model_name}

    if using_legacy_sdk():
        assert legacy_genai is not None
        return legacy_genai.GenerativeModel(
            model_name=model_name,
            generation_config=MODEL_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )

    raise RuntimeError("Unable to build model because no supported Gemini SDK is available.")


def _new_sdk_image_part(image_path: str):
    assert genai is not None
    image_info = image_format(image_path)[0]
    return genai.types.Part.from_bytes(
        data=image_info["data"],
        mime_type=image_info["mime_type"],
    )


def gemini_output(model, image_path: str, system_prompt: str, user_prompt: str) -> str:
    if isinstance(model, dict) and using_new_sdk():
        assert genai is not None
        client = model["client"]
        model_name = model["model_name"]
        
        # Strip "models/" prefix if present for new SDK
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        
        image_part = _new_sdk_image_part(image_path)
        
        # Try with text as string, image as Part
        response = client.models.generate_content(
            model=model_name,
            contents=[f"{system_prompt}\n\n{user_prompt}", image_part],
            config={
                "temperature": MODEL_CONFIG["temperature"],
                "top_p": MODEL_CONFIG["top_p"],
                "top_k": MODEL_CONFIG["top_k"],
                "max_output_tokens": MODEL_CONFIG["max_output_tokens"],
            },
        )
        return getattr(response, "text", "") or str(response)

    image_info = image_format(image_path)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    legacy_model = cast(Any, model)
    response = legacy_model.generate_content(input_prompt)
    return response.text


def run_notebook_equivalent_flow(model, image_path: str, system_prompt: str) -> None:
    # Equivalent to "EXTRACTING PART OF THE INFORMATION FROM INVOICE" cell.
    print("\n=== Balance Amount Query ===")
    balance_answer = gemini_output(model, image_path, system_prompt, DEFAULT_BALANCE_PROMPT)
    print(balance_answer)

    # Equivalent to "EXTRACTING WHOLE DATA IN JSON FROM INVOICE" cell.
    print("\n=== Full Invoice JSON Query ===")
    json_answer = gemini_output(model, image_path, system_prompt, DEFAULT_JSON_PROMPT)
    print(json_answer)


def format_gemini_error(exc: Exception) -> str:
    message = str(exc)
    upper_message = message.upper()

    if "RESOURCE_EXHAUSTED" in upper_message or " 429 " in f" {upper_message} ":
        return (
            "Gemini API quota exceeded (HTTP 429). "
            "Wait a bit and retry, or use a key/project with available quota."
        )

    if "NOT_FOUND" in upper_message and "MODEL" in upper_message:
        return (
            "Requested model is not available for this API version or key. "
            "Use --list-models to see valid model names and pass one via --model-name."
        )

    if "PERMISSION_DENIED" in upper_message or "UNAUTHENTICATED" in upper_message:
        return (
            "Authentication/permission issue with Gemini API key. "
            "Check GOOGLE_API_KEY or GEMINI_API_KEY in your environment/.env."
        )

    return f"Gemini request failed: {message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoice recognition using Gemini model (script version of the notebook)."
    )
    parser.add_argument(
        "--image-path",
        required=False,
        help="Path to the invoice image (png/jpeg/webp).",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-2.0-flash",
        help="Gemini model name. Use gemini-pro-vision only if enabled in your account.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all Gemini models that support generateContent.",
    )
    parser.add_argument(
        "--user-prompt",
        default="",
        help="Optional custom prompt. If empty, runs both notebook example prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional system prompt.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        unknown_text = " ".join(unknown).lower()
        if "python" in unknown_text and "invoice_recognition_gemini.py" in unknown_text:
            parser.error(
                "it looks like two commands were pasted together. "
                "Run one command at a time, or separate commands with ';' in PowerShell."
            )
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args


def main() -> None:
    args = parse_args()

    try:
        client = configure_gemini()

        if args.list_models:
            print("=== Supported Models ===")
            list_supported_models(client)
            return

        if not args.image_path:
            raise SystemExit("--image-path is required when not using --list-models")

        model = build_model(client, args.model_name)

        if args.user_prompt:
            output = gemini_output(model, args.image_path, args.system_prompt, args.user_prompt)
            print(output)
        else:
            run_notebook_equivalent_flow(model, args.image_path, args.system_prompt)
    except Exception as exc:
        raise SystemExit(format_gemini_error(exc)) from None


if __name__ == "__main__":
    main()
