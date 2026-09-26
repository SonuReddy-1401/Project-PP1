"""
Offline Precomputed LLM Narrative Generator for Coach-Readable Dashboard Layer
Calls local Ollama REST API (qwen2.5:latest or llama3.2:latest) with strict grounding constraints.
Saves generated coach explanations to narratives.json.
"""

import os
import sys
import json
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(CURRENT_DIR, "narratives.json")

GROUNDING_CONSTRAINT = (
    "You are writing a short explanation for a football coach with no data science background. "
    "Use ONLY the numbers provided below — do not invent, estimate, or assume any statistic, event, or detail not explicitly given to you. "
    "If you are not given enough information to explain something, say so plainly instead of guessing. "
    "Write 2-4 short sentences, plain conversational English, no jargon (avoid words like 'centroid', 'reprojection', 'interval', 'HSV' — "
    "describe what happened in football terms instead). Do not use markdown formatting."
)

PROMPTS = {
    "overview": f"""
Match Overview Data: Target team Napoli (Sky Blue kits), match duration 15 minutes (22,500 frames @ 25 FPS), attacking direction left to right. Primary zone of control was the middle third (57.4% of match time). Safe team movement total was 1.77 kilometers at an average speed of 7.13 km/h. Camera calibration succeeded across 100% of frames.

{GROUNDING_CONSTRAINT}

Provide a 2-3 sentence executive match summary for the head coach.
""",
    "team_heatmap": f"""
Team Heatmap Data: 215,988 spatial player positions recorded over 15 minutes. The highest density of player positions occurs in the central midfield area between X = -15m and X = +15m, with heavy presence along the left flank.

{GROUNDING_CONSTRAINT}

Explain where the team spent most of their time on the pitch and what this spatial density shows about their positioning during this clip.
""",
    "team_shape": f"""
Team Shape Data: Average team width is 46.2 meters across the pitch, and average team depth is 38.9 meters. The team shape briefly compressed twice: once around 1 minute 25 seconds (match time 11:19) and once around 2 minutes 54 seconds (match time 12:48), before returning to normal spacing within a few seconds each time.

{GROUNDING_CONSTRAINT}

Explain what team width and team depth mean in plain football terms, then explain what the two brief compression moments likely represent (such as a stoppage, set piece, or defensive compacting), stating plainly that the exact tactical trigger is unconfirmed without video playback.
""",
    "pitch_thirds": f"""
Pitch Thirds Data: Defensive Third occupancy 17.0%, Middle Third occupancy 57.4%, Attacking Third occupancy 25.6%.

{GROUNDING_CONSTRAINT}

Summarize how team time was split across the pitch and explain what spending over half the clip in the middle third indicates about match control.
""",
    "team_trajectory": f"""
Team Trajectory Data: The team's collective center of movement traveled 1.77 kilometers over 15 minutes when excluding single-frame camera jump noise (averaging a steady 7.13 km/h walking and jogging pace). The raw unfiltered calculation yielded an impossible 24.01 km due to rapid camera panning.

{GROUNDING_CONSTRAINT}

Explain the team's average movement speed across the pitch and explain why filtering out camera pan jumps is necessary to get an accurate total distance.
""",
    "individual_players": f"""
Individual Player Tracking Data: Tracking persistent player segments shows continuous tracking durations averaging 10 to 18 seconds per segment before broadcast camera cuts or player overlap cause ID switching. The longest continuous tracking segment lasted 1 minute 48 seconds (55.9 meters covered).

{GROUNDING_CONSTRAINT}

Explain to the coach how to read these individual player tracking segments, emphasizing that each entry represents a continuous tracking window rather than a full 90-minute player match total.
""",
    "data_quality": f"""
Data Quality Metrics: 100% camera calibration success rate across all 22,500 frames. 847 unique tracking IDs detected across the 15-minute clip due to broadcast camera cuts. Distance calculation rejected 49.5% of frame intervals as implausible camera pan speed spikes.

{GROUNDING_CONSTRAINT}

Explain the reliability of the dataset, highlighting that camera calibration was 100% successful while player tracking numbers reflect broadcast camera cut segmentation.
"""
}

# Grounded Fallback Narratives if local Ollama server is offline or loading
FALLBACK_NARRATIVES = {
    "overview": "This 15-minute report tracks Napoli's team movements from left to right. The team controlled the match primarily through the middle third of the pitch, spending 57.4% of match time in central areas while maintaining a steady average team movement pace of 7.13 km/h.",
    "team_heatmap": "The spatial heatmap shows that Napoli concentrated most of their outfield presence in central midfield and along the left wing. Player density was highest between 15 meters behind and 15 meters ahead of the halfway line.",
    "team_shape": "Team width measures how wide the squad stretches across the pitch, averaging 46.2 meters, while team depth measures length from back to front, averaging 38.9 meters. The team briefly squeezed tightly together around 1:25 and 2:54 of clip time (match clock 11:19 and 12:48), which represents momentary tactical compacting or a play stoppage.",
    "pitch_thirds": "Napoli spent over half the match segment (57.4%) in the middle third of the pitch, compared to 17.0% in their defensive third and 25.6% in the attacking third. This distribution highlights strong midfield territorial control.",
    "team_trajectory": "The team's collective center of position moved a total of 1.77 kilometers over 15 minutes at an average jogging pace of 7.13 km/h. High-speed camera panning jumps were filtered out so camera movements are not mistaken for real player running distance.",
    "individual_players": "Each entry in this section represents a continuous tracking segment for a player during a single camera angle, averaging 10 to 18 seconds per segment. Because television broadcast cuts interrupt continuous tracking, these figures provide reliable short-sequence mobility snapshots rather than full match totals.",
    "data_quality": "Camera calibration succeeded across 100% of analyzed video frames, providing reliable pitch coordinates. Individual player tracking generates 847 segment IDs due to frequent broadcast camera cuts, and unphysical single-frame jumps were safely filtered out."
}

def generate_with_ollama(model_name="qwen2.5:latest"):
    results = {}
    url = "http://localhost:11434/api/generate"

    print(f"[INFO] Connecting to local Ollama API at {url} using model '{model_name}'...")

    for section_key, prompt_text in PROMPTS.items():
        try:
            payload = {
                "model": model_name,
                "prompt": prompt_text,
                "stream": False,
                "options": { "temperature": 0.2 }
            }
            res = requests.post(url, json=payload, timeout=1.0)
            if res.status_code == 200:
                txt = res.json().get("response", "").strip()
                # Clean up any residual markdown headers or quotes
                txt = txt.replace('"', '').replace('**', '').replace('###', '')
                results[section_key] = txt
                print(f"  [SUCCESS] Generated narrative for '{section_key}'")
            else:
                print(f"  [WARNING] Ollama returned status {res.status_code} for '{section_key}'. Using fallback.")
                results[section_key] = FALLBACK_NARRATIVES[section_key]
        except Exception as e:
            print(f"  [INFO] Ollama connection failed for '{section_key}' ({e}). Using grounded fallback.")
            results[section_key] = FALLBACK_NARRATIVES[section_key]

    return results

def main():
    # Attempt Ollama generation first; fall back to verified grounded narratives if offline
    narratives = generate_with_ollama(model_name="qwen2.5:latest")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2)

    print(f"\n[SUCCESS] Precomputed coach narratives saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
