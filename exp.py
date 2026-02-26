import argparse
import os
import torch
from safetensors.torch import load_file
from rich.console import Console
from rich.table import Table

from e1_fastplms.modeling_e1 import E1ForMaskedLM


def load_weights(path, cast_fp32=True):
    assert os.path.exists(path), f"File {path} not found."
    if path.endswith(".safetensors"):
        sd = load_file(path)
    elif path.endswith(".pth") or path.endswith(".pt"):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
    else:
        try:
            sd = load_file(path)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=True)
    
    if cast_fp32:
        return {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default=None)
    parser.add_argument("--files", type=str, nargs="+", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--assert_exact", action="store_true")
    args = parser.parse_args()

    model = E1ForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', dtype=torch.float32).eval()
    torch.save(model.state_dict(), 'load_from_pretrained_1.pth')
    model = E1ForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', dtype=torch.float32).eval()
    torch.save(model.state_dict(), 'load_from_pretrained_2.pth')

    if args.file1 is None:
        args.file1 = 'load_from_pretrained_1.pth'
    if args.files is None:
        args.files = ['load_from_pretrained_2.pth']

    paths = [args.file1] + args.files
    sds = [load_weights(p, cast_fp32=not args.strict) for p in paths]
    all_keys = sorted(set().union(*(sd.keys() for sd in sds)))
    strict_mismatches = []

    console = Console()
    table = Table(title=f"Weights Comparison (Reference: {os.path.basename(paths[0])})")
    table.add_column("Tensor Name", style="cyan", no_wrap=True)
    
    for p in paths[1:]:
        table.add_column(f"{os.path.basename(p)} == Ref", justify="center")
    
    sd1 = sds[0]
    for k in all_keys:
        row = [k]
        
        has_ref = k in sd1
        ref_w = sd1[k] if has_ref else None
        
        for sd in sds[1:]:
            has_other = k in sd
            other_w = sd[k] if has_other else None
            
            if not has_ref or not has_other:
                if not has_ref and not has_other:
                    row.append("[dim]✔[/dim]")
                else:
                    row.append("[red]✘[/red]")
            else:
                # Both present, compare shapes and MSE
                assert isinstance(ref_w, torch.Tensor), f"Weight {k} in reference is not a tensor."
                assert isinstance(other_w, torch.Tensor), f"Weight {k} in comparison file is not a tensor."
                
                if ref_w.shape != other_w.shape:
                    row.append("[red]✘ (Shape)[/red]")
                else:
                    if args.strict:
                        if torch.equal(ref_w, other_w):
                            row.append("[green]✔[/green]")
                        else:
                            mse = torch.mean((ref_w.float() - other_w.float())**2).item()
                            row.append(f"[red]✘ (Strict, MSE: {mse:.2e})[/red]")
                            strict_mismatches.append(k)
                    else:
                        mse = torch.mean((ref_w - other_w)**2).item()
                        if mse == 0:
                            row.append("[green]✔[/green]")
                        else:
                            row.append(f"[red]✘ (MSE: {mse:.2e})[/red]")
        
        table.add_row(*row)

    console.print(table)
    if args.strict and args.assert_exact:
        assert len(strict_mismatches) == 0, (
            f"Found {len(strict_mismatches)} strict mismatches. "
            f"First mismatches: {strict_mismatches[:10]}"
        )


if __name__ == "__main__":
    main()