from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

for path in Path().rglob("*.yml"):
    with Path(path).open(mode="r") as f:
        data = yaml.load(f)

    with Path(path).open(mode="w") as f:
        yaml.dump(data, f)
