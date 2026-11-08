"""Create a self-contained service+vendor-SDK archive, without training code or model weights."""
import argparse
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=root / 'zerith_image_service/dist/zerith_h1_deploy.tar.gz')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as tar:
        for folder in ('zerith_image_service', 'zerith_H1_SDK1.3.6'):
            for path in sorted((root / folder).rglob('*')):
                if path.is_file() and not {'__pycache__', 'dist', 'spark'} & set(path.relative_to(root).parts) and path.suffix != '.pyc':
                    tar.add(path, arcname=str(path.relative_to(root)))
    print(args.output)


if __name__ == '__main__':
    main()
