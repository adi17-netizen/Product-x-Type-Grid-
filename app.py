import os
import re
import json
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
You are an expert startup analyst using the Product x Market Type framework by Sajith Pai.
Your job is to classify a startup into one of 9 cells on a 3x3 grid. You MUST classify
AT THE TIME OF THE STARTUP'S FOUNDING / INITIAL LAUNCH — not based on today's competitive
landscape. This is critical. Think about what the world looked like when this company first launched.

=== AXIS 1: MARKET TYPE (Competition at time of launch) ===

Apply these STRICT decision criteria from the author:

RED OCEAN:
- At least two DIRECT competitors with 10%+ market share each at time of launch
- Search volume for the category is meaningful enough to mount an SEO campaign
- Examples from the author: Lyft (Uber existed), Rapido 4W (Ola/Uber existed),
  Zoom preCOVID (Webex/Skype/GoToMeeting existed as DIRECT video call competitors),
  Groww (Zerodha/HDFC Securities existed), TikTok (Instagram/YouTube/Snapchat existed),
  Superhuman (Gmail/Outlook existed), Minimalist (many skincare brands existed)

ORANGE OCEAN:
- NO direct competitors doing the same thing, BUT indirect competitors exist
- The indirect competitors serve the need via a DIFFERENT mechanism or a subpar experience
- Negligible search volume for the specific product category
- Examples from the author: Stripe (PayPal/Authorize.net were indirect — payment processors
  but NOT developer-first API payments), PolicyBazaar (insurance agents were indirect — not
  online comparison), Airbnb (hotels/Craigslist were indirect — not peer-to-peer home sharing),
  Calendly (email back-and-forth was the indirect "competitor" — no direct scheduling link tool),
  Urban Company home salon (local parlours were indirect — not app-based home service),
  Sharechat (Facebook/Instagram existed but were English-first — no vernacular-first social app),
  Duolingo (language classes existed but not free gamified app-based learning)

BLUE OCEAN:
- NO direct OR indirect competitors. You are creating the need and defining the category.
- The category essentially didn't exist before this startup
- Examples from the author: Vanta (no one was productizing SOC2 compliance), GoKwik
  (no RTO risk scoring existed), PagerDuty (no incident alerting for engineering teams),
  Veeva (no pharma-specific CRM), Front (no shared inbox collaboration tool),
  Coinbase (no simple consumer crypto buying platform), Classplus (no coaching centre
  digital OS), Calm (no meditation app — created the entire category), CRED (no credit
  card bill payment rewards platform)

CRITICAL: "Direct competitor" means a company doing ESSENTIALLY THE SAME THING for the SAME
customer. PayPal is NOT a direct competitor to Stripe because PayPal was a consumer payment
platform, not a developer API. Gmail is a direct competitor to Superhuman because both are
email clients. Hotels are NOT direct competitors to Airbnb because they are a fundamentally
different model (commercial lodging vs peer-to-peer home sharing).

=== AXIS 2: PRODUCT TYPE (Pain level) ===

Apply these STRICT decision criteria from the author:

PAINKILLER: Must meet at least one of:
  a) ICP would be happy to pay to solve this TODAY, and/or
  b) ICP is currently using a MANUAL WORKAROUND to solve the problem
- There is hair-on-fire urgency. "For a hair on fire customer, even a blanket is a fire extinguisher."
- Examples: Stripe (developers wasting days on payment integration), Vanta (companies desperately
  needing SOC2), PolicyBazaar (people being missold insurance), Lyft/Rapido (people needing rides),
  Airbnb (people needing affordable accommodation)

TOOTHPICK: "Not solving a deep pain, but solving for something irritating, like a small particle
of food stuck in your teeth. You can get through the event/meeting, but you are forever fiddling
at it with your tongue."
- Users complain about the problem but have NOT actively sought a dedicated solution
- Examples: Zoom preCOVID (video calls worked via Webex/Skype — annoying but functional),
  Groww (investing was possible via existing platforms — just intimidating for beginners),
  Calendly (scheduling was doable via email — just tedious),
  Urban Company (going to a salon was possible — just inconvenient with kids),
  Coinbase (buying crypto was possible — just sketchy and intimidating),
  Classplus (coaching centres ran on WhatsApp — functional but limiting)

VITAMIN: "Nice to have but can do without it. Will only pay for it after they experience the value."
- No urgency whatsoever. Success requires creating a NEW habit or behaviour.
- Examples: Calm (meditation app — nobody needed this), CRED (credit card rewards — pure nice-to-have),
  TikTok/Minimalist (entertainment/skincare — vitamin categories),
  Duolingo (casual language learning — nice to have), Sharechat (vernacular social — nice to have),
  Superhuman (faster email — Gmail works fine for 99% of people)

CRITICAL CALIBRATION: Most products are NOT painkillers. The author explicitly says "Painkiller x
Blue Ocean is the rarest of the nine categories" and that "most products fall in the Toothpick x
Orange Ocean segment." If you find yourself classifying everything as Painkiller x Red Ocean, you
are doing it wrong. Ask yourself: "Is this truly a hair-on-fire problem where people are desperately
using manual workarounds? Or is it merely annoying/inconvenient (Toothpick) or nice-to-have (Vitamin)?"

=== THE 9 CELLS AND WHAT THEY MEAN ===

| Product Type  | Blue Ocean           | Orange Ocean         | Red Ocean            |
|---------------|----------------------|----------------------|----------------------|
| Painkiller    | ETP (rarest, best)   | ETP (very attractive)| EUP (wedge needed)   |
| Toothpick     | ELP (must delight)   | EUP/ELP (common)     | EUP (wedge needed)   |
| Vitamin       | ELP (create habit)   | Full Product (tough)  | Full Product (hardest)|

LAUNCH STRATEGIES:
- ETP (Earliest Testable Product): Barebones product delivering atomic unit of value. Just enough
  to validate. Vanta launched with a spreadsheet. Stripe launched US-only, no subscriptions.
- EUP (Earliest Usable Product): More elaborate than ETP. Must have default category features plus
  differentiating features for the underserved segment. Classic wedge strategy.
- ELP (Earliest Lovable Product): Must delight from day one. Find users who obsess over your features.
  Create new behaviour. Product has to be beautiful and habit-forming.
- Full Product: Near-complete, polished product. Need to match incumbents and add delight features.
  Hardest path — reserved for Vitamin products in competitive markets.

=== PLAYBOOKS BY CATEGORY ===

Painkiller x Blue/Orange Ocean:
- Playbook: Launch narrow product delivering atomic unit of value. Validate problem + solution.
- VC Pitch: Show customer desperation — workarounds, willingness to pay, examples of pull.
- Mistake to avoid: Overbuilding the product before validation.

Painkiller x Red Ocean AND Toothpick x Red Ocean:
- Playbook: Find underserved/unserved segment incumbents can't or won't serve. Build wedge product.
  Differentiate on features that matter to this segment. Expand from beachhead.
- VC Pitch: Show the underserved segment, their low NPS, your unique wedge/attack angle.
- Mistake to avoid: Frontal assault on incumbent instead of wedge approach.

Toothpick x Blue Ocean AND Vitamin x Blue Ocean:
- Playbook: Must delight from day one. Segment market to find users who treat it as a Painkiller.
  Create new behaviour/habit. Product must be lovable, not just functional.
- VC Pitch: Show the segment for whom this is a true Painkiller + expansion playbook.
- Mistake to avoid: Shipping too early without segmenting to find obsessed users.

Toothpick x Orange Ocean:
- Playbook: Find segment for whom this is a Painkiller. Launch with EUP (B2B) or ELP (B2C).
  Adopt wedge strategy against indirect competitors.
- VC Pitch: Show power user segment + underserved dynamics making it like a Blue Ocean.
- Mistake to avoid: Shipping too early, not adopting wedge strategy.

Vitamin x Orange/Red Ocean:
- Playbook: Need full product with delight features. Find narrow power users underserved by
  incumbents. Requires strong founder-problem fit and unique attack angle.
- VC Pitch: Show underserved power user, unique insight/attack angle, strong why-now.
- Mistake to avoid: Shipping without delight features. This is the hardest category.

=== STEP-BY-STEP CLASSIFICATION PROCESS ===

You MUST follow these steps in order:

STEP 1 - TEMPORAL CONTEXT: Determine WHEN this startup launched. All classification must be
based on the competitive landscape AT THAT TIME, not today.

STEP 2 - MARKET TYPE: List all competitors that existed at launch time. For each, determine
if they were DIRECT (doing essentially the same thing for the same customer) or INDIRECT
(serving the need via a different mechanism). Apply the strict Red/Orange/Blue criteria.

STEP 3 - PRODUCT TYPE: Ask yourself these questions in order:
  a) At launch, was the ICP using manual workarounds? Would they pay immediately? → Painkiller
  b) Were users complaining but not actively seeking solutions? Annoyed but coping? → Toothpick
  c) Was this purely nice-to-have, creating a new behaviour? → Vitamin

STEP 4 - CROSS-CHECK against the author's examples. If your classification puts a company in a
very different cell than a similar company the author classified, reconsider your reasoning.

---

Here is information about the startup to analyze:

{startup_info}

---

Classify this startup following the exact step-by-step process above. Be rigorous. Most startups
are NOT Painkillers — really interrogate whether the problem is truly hair-on-fire or merely
annoying. Most markets are NOT Red Oceans — really interrogate whether competitors are truly
DIRECT or merely indirect.

Keep reasoning fields concise — 2-3 sentences max each.

Return ONLY a valid JSON object (no markdown, no code blocks) with this exact structure:
{{
  "company_name": "string",
  "tagline": "one-line description of what they do",
  "founding_year": "approximate year of founding/launch",
  "sector": "e.g. Fintech / Payments, Edtech, SaaS, etc.",
  "hq": "e.g. San Francisco, CA or Bangalore, India",
  "product_type": "Painkiller" | "Toothpick" | "Vitamin",
  "market_type": "Blue Ocean" | "Orange Ocean" | "Red Ocean",
  "cell": "e.g. Toothpick x Orange Ocean",
  "launch_strategy": "ETP" | "EUP" | "ELP" | "Full Product",
  "launch_strategy_name": "Full name — one of: Earliest Testable Product, Earliest Usable Product, Earliest Lovable Product, Full-blown Product",
  "product_confidence": 0-100,
  "market_confidence": 0-100,
  "product_reasoning": "2-3 sentences max. First state what problem the startup solves. Then apply the strict Painkiller/Toothpick/Vitamin criteria with specific evidence. Explain WHY it is or isn't a Painkiller — was there true hair-on-fire urgency and manual workarounds, or was it more of an annoyance or nice-to-have?",
  "market_reasoning": "2-3 sentences max. First list the competitors at time of launch. Then for each, state whether they were DIRECT or INDIRECT and why. Then apply the strict Red/Orange/Blue criteria to reach your conclusion.",
  "key_insight": "The single most important strategic insight about this startup's position in the framework",
  "playbook": "Specific actionable advice based on their cell, referencing the framework's recommended approach",
  "vc_pitch": "How they should pitch to VCs given their specific category",
  "founder_mistake": "The specific noob founder mistake to avoid for their category",
  "competitors": ["list", "of", "competitors", "at", "launch"],
  "competitor_type": "direct" | "indirect" | "none",
  "wedge_opportunity": "The underserved segment or attack angle they can exploit"
}}
"""


MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-001"]

def analyze_with_gemini(url: str) -> dict:
    # Let Gemini's own search grounding do all the research — no Jina pre-fetch needed
    is_url = url.startswith("http")
    prompt = FRAMEWORK_PROMPT.format(startup_info=f"Startup: {url}")
    if is_url:
        full_prompt = f"Research the startup at {url} using Google Search, understand their business, founding year, competitors at launch, and product. Then classify them per the framework below.\n\n{prompt}"
    else:
        full_prompt = f"Research the startup called '{url}' using Google Search, understand their business, founding year, competitors at launch, and product. Then classify them per the framework below.\n\n{prompt}"

    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
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
    query = req.url.strip()
    # If it looks like a URL (has a dot and no spaces), treat as URL
    # Otherwise treat as a plain company name and let Gemini search for it
    if not query.startswith("http") and ("." in query and " " not in query):
        query = "https://" + query

    try:
        result = analyze_with_gemini(query)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result
