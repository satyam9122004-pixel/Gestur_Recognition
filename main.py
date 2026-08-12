"""Entry point for the person identification system."""
import argparse

from integration.pipeline import IdentificationPipeline
from database.register import register_person


def main():
    parser = argparse.ArgumentParser(description="Classical-CV person identification system.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the live identification pipeline")
    run_parser.add_argument("--video", default=0, help="Video source (camera index or file path)")
    run_parser.add_argument("--db", default="database/people.db", help="Path to the SQLite database file")
    run_parser.add_argument("--threshold", type=float, default=15.0, help="Match distance threshold")
    run_parser.add_argument("--no-display", action="store_true", help="Disable the live video window")

    register_parser = subparsers.add_parser("register", help="Register a new person")
    register_parser.add_argument("--name", required=True, help="Name of the person to register")
    register_parser.add_argument("--video", default=0, help="Video source (camera index or file path)")
    register_parser.add_argument("--samples", type=int, default=30, help="Number of feature samples to collect")
    register_parser.add_argument("--db", default="database/people.db", help="Path to the SQLite database file")

    args = parser.parse_args()

    if args.command == "run":
        video_source = int(args.video) if str(args.video).isdigit() else args.video
        pipeline = IdentificationPipeline(
            db_path=args.db,
            video_source=video_source,
            threshold=args.threshold,
            display=not args.no_display,
        )
        pipeline.run()

    elif args.command == "register":
        video_source = int(args.video) if str(args.video).isdigit() else args.video
        register_person(args.name, video_source=video_source, num_samples=args.samples, db_path=args.db)


if __name__ == "__main__":
    main()
