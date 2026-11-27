from . import audio_utils
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# -------------------------------------------------- ⬇ Save route (no re-run)
# The frontend POSTs the current trim parameters; the server trims the temp
# mirror that the last run wrote (the exact audio the timeline shows) and
# writes a FLAC into ComfyUI's output folder. This avoids re-running the
# upstream workflow just to persist a cut.
try:
    from aiohttp import web
    from server import PromptServer

    _routes = PromptServer.instance.routes

    @_routes.post("/audio_trim_visual/save")
    async def zat_save_trim(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON body"}, status=400)
        try:
            import folder_paths

            info = audio_utils.save_trimmed_flac(
                str(body.get("mirror", "")),
                folder_paths.get_temp_directory(),
                folder_paths.get_output_directory(),
                float(body.get("start", 0.0)),
                float(body.get("end", 0.0)),
                float(body.get("gain_db", 0.0)),
                str(body.get("filename_prefix", "")),
            )
            return web.json_response({"ok": True, **info})
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": str(exc)}, status=400)
except Exception:
    # No PromptServer (tests / plain import): the node itself still works,
    # only the ⬇ Save route is unavailable.
    pass
