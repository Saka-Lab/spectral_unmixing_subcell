import httpx
import shutil
import typer
import zipfile
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

app = typer.Typer(help="Download and unpack files from a restricted Zenodo record.")
console = Console()

ZENODO_API = "https://zenodo.org/api/records"


def fetch_record_metadata(record_id: str, token: str) -> dict:
    """Fetch record metadata from the Zenodo API."""
    url = f"{ZENODO_API}/{record_id}"
    params = {"token": token}
    with httpx.Client(follow_redirects=True) as client:
        resp = client.get(url, params=params)
    if resp.status_code == 404:
        console.print(f"[red]Record {record_id} not found.[/red]")
        raise typer.Exit(1)
    if resp.status_code == 403:
        console.print("[red]Access denied. The share token may be invalid or expired.[/red]")
        raise typer.Exit(1)
    resp.raise_for_status()
    return resp.json()


def download_file(url: str, token: str, dest: Path) -> None:
    """Stream-download a single file with a progress bar."""
    params = {"token": token}
    with httpx.Client(follow_redirects=True) as client:
        with client.stream("GET", url, params=params) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]{dest.name}", total=total or None)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))


@app.command()
def download_and_unzip(
    record_id: str = typer.Option(
        ...,
        "--record-id", "-r",
        help="Zenodo record ID.",
    ),
    token: str = typer.Option(
        ...,
        "--token", "-t",
        help="Zenodo share token for restricted access.",
    ),
    dest: Path = typer.Option(
        Path("."),
        "--dest", "-d",
        help="Destination root where zip contents will be extracted.",
    ),
    keep_zip: bool = typer.Option(
        False,
        "--keep-zip",
        help="Keep the zip file(s) after unpacking.",
    ),
):
    """Download files from a Zenodo record and unpack zip contents to dest."""
    console.print(f"[bold]Fetching metadata for record [cyan]{record_id}[/cyan]...[/bold]")
    metadata = fetch_record_metadata(record_id, token)
    title = metadata.get("metadata", {}).get("title", "Unknown title")
    console.print(f"[green]Record:[/green] {title}\n")

    files = metadata.get("files", [])
    if not files:
        console.print("[yellow]No files found in this record.[/yellow]")
        raise typer.Exit()

    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_tmp_unpack"

    for f in files:
        filename = f["key"]
        file_url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        if not file_url:
            console.print(f"[yellow]Skipping {filename}: no download URL.[/yellow]")
            continue

        zip_path = dest / filename
        if not zip_path.exists():
            console.print(f"[bold]Downloading {filename}...[/bold]")
            download_file(file_url, token, zip_path)
            console.print(f"[green]✓ Downloaded {filename}[/green]\n")
        else:
            console.print(f"[yellow]Skipping download, {filename} already exists.[/yellow]")

        if filename.endswith(".zip"):
            console.print(f"[bold]Unpacking {filename}...[/bold]")
            tmp.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)

            for item in tmp.iterdir():
                for subitem in item.iterdir():
                    target = dest / subitem.name
                    if target.exists():
                        console.print(f"  [yellow]Removing existing {target}[/yellow]")
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                    shutil.move(str(subitem), dest)
                    console.print(f"  [green]✓ Moved {item.name} → {dest}[/green]")

            shutil.rmtree(tmp)

            if not keep_zip:
                zip_path.unlink()
                console.print(f"  [dim]Deleted {filename}[/dim]")

    console.print("\n[bold green]All done![/bold green]")


if __name__ == "__main__":
    app()