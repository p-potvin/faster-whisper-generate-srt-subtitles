
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

def print_progress(current, total, prefix="", width=30):
    ratio = current / total if total > 0 else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total}\r", end="", flush=True)
    if current >= total:
        print("", flush=True)

def write_srt(output_path, segments, texts):
    with open(output_path, "w", encoding="utf-8") as output_file:
        for segment, text in zip(segments, texts):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            segment_id = segment.id + 1
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            output_file.write(line_out)
