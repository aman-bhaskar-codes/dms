"""Supported DMS CLI."""

import argparse
import asyncio
import json
from core.config import Settings


def cli():
    parser = argparse.ArgumentParser(description="DMS — local supervised demonstration")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Start the local dashboard")
    run.add_argument(
        "--replay", action="store_true", help="Automatically start the synthetic replay"
    )
    check = sub.add_parser("check", help="Check local setup without contacting Gemini")
    check.add_argument("--probe-camera", action="store_true")
    models = sub.add_parser("setup-models", help="Explicitly install and verify local models")
    models.add_argument("--knowledge", action="store_true")
    models.add_argument("--voice", action="store_true")
    legacy = sub.add_parser("import-legacy", help="Read-only import from a legacy database")
    legacy.add_argument("path")
    args = parser.parse_args()
    config = Settings()
    saved = config.data_dir / "settings.json"
    if saved.is_file():
        config = Settings(**(config.model_dump() | json.loads(saved.read_text())))
    if args.command == "check":
        from core.setup import check_setup

        print(json.dumps(check_setup(config, args.probe_camera), indent=2))
    elif args.command == "setup-models":
        from core.setup import setup_models

        print(json.dumps(setup_models(config, args.knowledge, args.voice), indent=2))
    elif args.command == "import-legacy":
        from memory.store import Store

        async def migrate():
            store = Store(config.db_path)
            await store.connect(recover_sessions=False)
            try:
                print(json.dumps({"imported": await store.import_legacy(args.path)}))
            finally:
                await store.close()

        asyncio.run(migrate())
    else:
        import uvicorn
        from ui.web.server import create_app

        app = create_app(config)
        app.state.autostart = "replay" if getattr(args, "replay", False) else None
        uvicorn.run(
            app,
            host=config.web_host,
            port=config.web_port,
            log_level="warning",
            timeout_graceful_shutdown=5,
            ws_max_size=8192,
        )


if __name__ == "__main__":
    cli()
