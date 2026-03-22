import os
import re
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class AnalyzeRequest(BaseModel):
    url: str


FRAMEWORK_PROMPT = """
You are an expert startup analyst. Your job is to classify a startup using the Product x Market Type framework by Sajith Pai.

THE FRAMEWORK:

TWO AXES:

1. MARKET TYPE (Competition):
   - Blue Ocean: No direct or indirect competitors. Creating the need and defining the category.
   - Orange Ocean: No direct competitors, but indirect competitors exist (e.g., Stripe had PayPal/Authorize.net, Airbnb had hotels/Craigslist). Negligible direct search volume.
   - Red Ocean: At least two direct competitors with 10%+ market share each. Meaningful search volume for the category.

2. PRODUCT TYPE (Pain level):
   - Painkiller: ICP would happily pay to solve this today. They are using manual workarounds. Hair-on-fire urgency.
   - Toothpick: A niggle or irritation — not an acute pain, but something that bothers you. Users complain but haven't actively sought a solution.
   - Vitamin: Nice to have. No urgency. Users will only pay after experiencing value. Habit-forming products.

THE 9 CELLS AND THEIR LAUNCH STRATEGY:
- Painkiller x Blue Ocean → ETP (Earliest Testable Product) — rarest, best category
- Painkiller x Orange Ocean → ETP — very attractive, address urgent pain vs indirect competition
- Painkiller x Red Ocean → EUP (Earliest Usable Product) — need wedge into underserved segment
- Toothpick x Red Ocean → EUP — need wedge, competitive but clear demand signal
- Toothpick x Blue Ocean → ELP (Earliest Lovable Product) — must delight, no urgency
- Vitamin x Blue Ocean → ELP — creating new behavior/category
- Toothpick x Orange Ocean → EUP or ELP — find segment for whom it's a painkiller
- Vitamin x Orange Ocean → Full Product — tough category, need full product + wedge
- Vitamin x Red Ocean → Full Product — hardest category, need strong founder-problem fit

PLAYBOOKS:
- ETP: Launch narrow product delivering atomic unit of value. Validate problem + solution.
- EUP: Match default features of category + differentiating features for underserved segment. Classic wedge.
- ELP: Must delight from day one. Find obsessed users (who treat it as a painkiller). Create new behavior.
- Full Product: Need complete, polished product. Find niche power users underserved by incumbents.

VC PITCH ANGLES:
- Painkiller x Blue/Orange Ocean: Show customer desperation — workarounds, willingness to pay
- Painkiller/Toothpick x Red Ocean: Show the underserved segment, their low NPS, your wedge
- Toothpick/Vitamin x Blue Ocean: Show the segment who treats it as a painkiller + expansion playbook
- Vitamin x Orange/Red Ocean: Show underserved power users, unique attack angle, strong why-now

NOOB FOUNDER MISTAKES:
- Painkiller x Blue/Orange Ocean: Overbuilding
- Painkiller/Toothpick x Red Ocean: Frontal assault on incumbent instead of wedge
- Toothpick/Vitamin x Blue Ocean: Shipping too early without finding obsessed users
- Toothpick x Orange Ocean: Shipping too early without wedge strategy
- Vitamin x Orange/Red Ocean: Shipping without delight features

---

Here is information about the startup to analyze:

{startup_info}

---

Based on all the above, classify this startup. Be analytical, specific, and cite evidence from the startup's actual business.

Return ONLY a valid JSON object (no markdown, no code blocks) with this exact structure:
{{
  "company_name": "string",
  "tagline": "one-line description of what they do",
  "product_type": "Painkiller" | "Toothpick" | "Vitamin",
  "market_type": "Blue Ocean" | "Orange Ocean" | "Red Ocean",
  "cell": "e.g. Painkiller x Orange Ocean",
  "launch_strategy": "ETP" | "EUP" | "ELP" | "Full Product",
  "product_confidence": 0-100,
  "market_confidence": 0-100,
  "product_reasoning": "2-3 sentences explaining the product classification with specific evidence",
  "market_reasoning": "2-3 sentences explaining the market classification with specific evidence",
  "key_insight": "The single most important strategic insight about this startup's position",
  "playbook": "What they should do — specific actionable advice based on their cell",
  "vc_pitch": "How they should pitch to VCs given their category",
  "founder_mistake": "The key mistake to avoid given their category",
  "competitors": ["list", "of", "known", "competitors"],
  "wedge_opportunity": "The underserved segment or attack angle they can exploit"
}}
"""


def fetch_startup_info(url: str) -> str:
    info_parts = []

    # Fetch homepage via Jina Reader
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(jina_url, timeout=15, headers={"Accept": "text/plain"})
        if resp.status_code == 200:
            text = resp.text[:4000]
            info_parts.append(f"HOMEPAGE CONTENT:\n{text}")
    except Exception:
        pass

    return "\n\n".join(info_parts) if info_parts else f"Startup URL: {url}"


MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-lite"]

def analyze_with_gemini(startup_info: str, url: str) -> dict:
    prompt = FRAMEWORK_PROMPT.format(startup_info=startup_info)
    full_prompt = f"Research the startup at {url} thoroughly using Google Search, then analyze it.\n\n{prompt}"

    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            last_error = e
            err_str = str(e)
            # Only fall through to next model on quota errors
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                continue
            raise

    raise HTTPException(
        status_code=429,
        detail="Gemini API quota exceeded on all available models. Please enable billing at aistudio.google.com or wait and try again."
    )


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    startup_info = fetch_startup_info(url)

    try:
        result = analyze_with_gemini(startup_info, url)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
