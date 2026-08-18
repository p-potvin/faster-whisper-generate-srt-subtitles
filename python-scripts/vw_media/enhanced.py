"""Output routing for the enhanced copy.

The original file is never modified. Enhancements accumulate on a copy in a
hidden ``.enhanced/`` sibling directory, keyed by the original filename, so a
video that has had both audio and video enhancement applied has exactly one
enhanced copy carrying both — and the sidecar records which passes produced it.

``--output`` overrides the destination entirely, for callers that want the
result somewhere specific (batch jobs, the CLI, tests). State is still tracked
against the original.
"""

import os
import shutil
import time


def enhanced_dir_for(video_path):
    return os.path.join(os.path.dirname(os.path.abspath(video_path)), '.enhanced')


def default_output_path(video_path):
    """Where an enhanced copy of *video_path* lives by default."""
    return os.path.join(enhanced_dir_for(video_path), os.path.basename(video_path))


def resolve_output_path(video_path, output_override=None):
    """Resolve the final output path, honouring an explicit ``--output``.

    A directory-valued override keeps the original basename inside it.
    """
    if not output_override:
        return default_output_path(video_path)

    out = os.path.abspath(os.path.expanduser(output_override))
    if os.path.isdir(out) or output_override.endswith((os.sep, '/')):
        return os.path.join(out, os.path.basename(video_path))
    return out


def temp_path_for(output_path):
    base, ext = os.path.splitext(output_path)
    return f"{base}.tmp{ext}"


def make_temp_dir(output_path, prefix="work"):
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    temp_dir = os.path.join(parent, f".{prefix}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def prepare_output(video_path, output_override=None):
    """Create the output directory and clear any stale temp file.

    Returns ``(output_path, temp_output_path)``.
    """
    output_path = resolve_output_path(video_path, output_override)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_output_path = temp_path_for(output_path)
    if os.path.exists(temp_output_path):
        try:
            os.remove(temp_output_path)
        except Exception as err:
            print(f"[enhanced] Could not remove stale temp output {temp_output_path}: {err}")
    return output_path, temp_output_path


def resolve_source(video_path, output_path, temp_dir):
    """Pick the input for this pass: the existing enhanced copy if there is one.

    Enhancements stack, so a second action reads the output of the first. The
    copy into *temp_dir* is what lets us write the new result back to
    *output_path* without reading and writing the same file at once.
    """
    if not os.path.exists(output_path):
        return video_path

    print("[enhanced] Applying on top of the existing enhanced copy...")
    staged = os.path.join(temp_dir, "source" + os.path.splitext(video_path)[1])
    try:
        shutil.copy2(output_path, staged)
        return staged
    except Exception as err:
        print(f"[enhanced] Could not stage existing enhanced copy ({err}); using the original")
        return video_path


def promote(temp_output_path, output_path):
    """Atomically make the freshly encoded file the enhanced copy.

    Nothing becomes visible until the encode has actually succeeded, so an
    interrupted run can never leave a half-written video looking complete.
    """
    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
        raise RuntimeError("Encoding produced no output file")
    try:
        os.replace(temp_output_path, output_path)
    except Exception as err:
        raise RuntimeError(f"Failed to promote temp output to {output_path}: {err}")
    return output_path


def discard_temp(*paths):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            pass
