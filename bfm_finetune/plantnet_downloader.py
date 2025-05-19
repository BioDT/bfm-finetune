import requests
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(view_url: str, out_path: str):
    page = requests.get(view_url)
    page.raise_for_status()
    text = page.text
    lines = text.split("\n")
    with_raw_path = next(l for l in lines if "rawPath" in l)
    url_raw = with_raw_path.split("'")[1]
    url = url_raw.encode().decode("unicode-escape")
    print(f"downloading {url}")
    res = requests.get(url)
    with open(out_path, "wt") as f:
        f.write(res.text)


if __name__ == "__main__":
    app()
