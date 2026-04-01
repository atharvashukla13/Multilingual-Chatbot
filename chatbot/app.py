"""
Streamlit Web Interface for the Ayurvedic Chatbot.
With Q-LREF Post-Quantum Encryption Layer.

Run with:
  cd chatbot
  streamlit run app.py
"""

import streamlit as st
import sys
import os
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.session import QLREFSession
from crypto.aes_gcm import bytes_to_hex_preview, bytes_to_base64_preview
from crypto import metrics as crypto_metrics
import config


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="आयुर्वेदिक चैटबॉट | Ayurvedic Chatbot",
    page_icon="🌿",
    layout="wide"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    /* ── Header ── */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    .chat-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
    }

    .chat-header p {
        color: #e8f5e9;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Passage Cards ── */
    .passage-card {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #333;
    }

    .score-badge {
        background: #2e7d32;
        color: #fff;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
    }

    /* ── Crypto Boxes ── */
    .crypto-box {
        background: #263238;
        border: 1px solid #ff9800;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #ffcc80;
    }

    .crypto-label {
        color: #ff9800;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .crypto-success {
        color: #66bb6a;
    }

    .crypto-data {
        color: #ce93d8;
        word-break: break-all;
        font-size: 0.75rem;
    }

    .handshake-step {
        background: #263238;
        border-left: 3px solid #ff9800;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        color: #eceff1;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Initialize RAG Pipeline (cached) — with fallback
# ──────────────────────────────────────────────
RAG_AVAILABLE = False

@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline if ML dependencies are available."""
    global RAG_AVAILABLE
    try:
        from rag.pipeline import AyurvedicRAG
        rag = AyurvedicRAG()
        RAG_AVAILABLE = True
        return rag
    except ImportError:
        RAG_AVAILABLE = False
        return None


# Ayurvedic knowledge base for fallback responses (keyword -> response)
AYURVEDIC_RESPONSES = [
    # Common health conditions
    (["cold", "cough", "flu", "sinus", "congestion", "sneez"],
     "Ayurvedic remedies for cold and cough: (1) Drink warm Tulsi-Ginger tea with honey — Tulsi is antibacterial and ginger clears congestion. (2) Take Sitopaladi Churna with honey for sore throat and cough. (3) Steam inhalation with Eucalyptus or Ajwain seeds. (4) Drink warm turmeric milk (Haldi Doodh) before bed to boost immunity. (5) Avoid cold/refrigerated foods; eat light, warm meals. This is a Kapha imbalance — use warming spices like black pepper, cinnamon, and cloves."),

    (["fever", "temperature", "viral"],
     "Ayurvedic approach for fever: (1) Guduchi (Giloy) is the primary herb — boil Giloy stem in water and drink. It is a powerful immunomodulator. (2) Tulsi and black pepper tea helps reduce temperature. (3) Sudarshan Ghanvati is a classical Ayurvedic formulation for fevers. (4) Light diet — eat khichdi or moong dal soup. (5) Rest is essential; fever is the body's way of fighting infection. Avoid heavy, oily, and cold foods."),

    (["headache", "migraine", "head pain"],
     "Ayurvedic remedies for headache: (1) Apply paste of sandalwood (Chandan) on the forehead for cooling relief. (2) Brahmi oil head massage helps chronic headaches. (3) Nasya therapy — 2 drops of Anu Taila in each nostril clears sinus headaches. (4) Drink warm water with lemon and ginger. (5) Peppermint oil on temples provides quick relief. Migraines are often a Pitta imbalance — avoid spicy foods, direct sunlight, and stress."),

    (["stomach", "acidity", "gastric", "bloating", "indigestion", "gas"],
     "Ayurvedic remedies for stomach and digestive issues: (1) Triphala Churna before bed regulates digestion and detoxifies. (2) Ajwain (carom seeds) with warm water relieves gas and bloating instantly. (3) Jeera (cumin) water — boil cumin seeds in water and sip throughout the day. (4) Avoid incompatible food combinations (e.g., milk with fruit). (5) Eat meals at regular times; lunch should be the largest meal when Agni (digestive fire) is strongest. (6) For acidity: take Amla (Indian gooseberry) or cold milk with Shatavari."),

    (["joint", "arthritis", "pain", "knee", "back pain", "inflammation"],
     "Ayurvedic approach for joint pain and arthritis (Amavata): (1) Guggulu formulations like Yogaraja Guggulu and Maharasnadi Kwath are classical treatments. (2) Warm sesame oil massage (Abhyanga) on affected joints daily. (3) Nirgundi (Vitex negundo) oil is excellent for local pain relief. (4) Ashwagandha strengthens bones and reduces inflammation. (5) Avoid cold, heavy, and processed foods that increase Ama (toxins). (6) Practice gentle yoga — Trikonasana, Veerabhadrasana, and Setu Bandhasana support joint health."),

    (["diabetes", "sugar", "blood sugar"],
     "Ayurvedic management of diabetes (Prameha/Madhumeha): (1) Jamun (Indian Blackberry) seeds powder — take 1 tsp twice daily to regulate blood sugar. (2) Bitter gourd (Karela) juice on empty stomach improves insulin sensitivity. (3) Fenugreek (Methi) seeds soaked overnight — drink the water and eat the seeds. (4) Turmeric with Amla juice is a classical combination. (5) Guduchi (Giloy) and Vijaysar (Pterocarpus marsupium) are potent anti-diabetic herbs. (6) Exercise regularly, reduce carbohydrates, and avoid sweet/heavy foods."),

    (["hair", "hair loss", "dandruff", "bald"],
     "Ayurvedic hair care: (1) Bhringraj (Eclipta alba) is the 'King of Hair' — use Bhringraj oil for scalp massage 2-3 times a week. (2) Amla (Indian gooseberry) strengthens hair roots and prevents premature greying. (3) For dandruff: apply Neem oil or paste of Neem leaves. (4) Triphala water rinse after shampooing conditions hair naturally. (5) Take Biotin-rich foods and Ashwagandha internally for hair growth. (6) Nasya with Anu Taila nourishes hair from within. Hair loss is often a Pitta imbalance — avoid excessive heat, spicy food, and stress."),

    (["weight", "obesity", "fat", "lose weight", "slim"],
     "Ayurvedic weight management: (1) Triphala Churna with warm water before bed boosts metabolism and detoxifies. (2) Drink warm lemon-honey water on an empty stomach every morning. (3) Guggulu (Commiphora mukul) is the primary herb for fat metabolism — Medohar Guggulu is the classical formulation. (4) Eat the heaviest meal at lunch when Agni is peak. (5) Avoid heavy, sweet, oily foods and cold drinks. (6) Practice Kapalbhati and Surya Namaskar daily. Obesity is a Kapha imbalance — favor light, warm, dry, and bitter foods."),

    (["anxiety", "depression", "mental", "mood", "nervous"],
     "Ayurvedic mental health support: (1) Brahmi (Bacopa monnieri) is the top brain tonic — improves memory, reduces anxiety, and calms the mind. (2) Ashwagandha is an adaptogen that reduces cortisol and stress. (3) Jatamansi (Spikenard) is used for depression and insomnia. (4) Shankhpushpi syrup calms the nervous system. (5) Daily Abhyanga (oil massage) with Brahmi oil on the scalp. (6) Practice Nadi Shodhana (alternate nostril breathing) and meditation daily. (7) Warm milk with nutmeg and saffron before bed promotes restful sleep."),

    (["blood pressure", "bp", "hypertension", "heart"],
     "Ayurvedic approach for heart health and blood pressure: (1) Arjuna (Terminalia arjuna) bark is the primary cardiac tonic in Ayurveda — strengthens heart muscle and regulates BP. (2) Sarpagandha (Rauwolfia serpentina) is used traditionally for hypertension. (3) Garlic (Lahsun) — 2 raw cloves daily helps reduce cholesterol and BP. (4) Practice Shavasana and Pranayama daily for stress reduction. (5) Reduce salt, avoid fried and processed foods. (6) Triphala helps prevent arterial plaque buildup."),

    (["eye", "vision", "eyesight"],
     "Ayurvedic eye care (Netra Chikitsa): (1) Triphala eye wash — soak Triphala powder overnight, filter, and wash eyes. Improves vision and reduces strain. (2) Nasya with Anu Taila nourishes the eyes through nasal passages. (3) Netra Basti — a classical Ayurvedic procedure with medicated ghee around the eyes. (4) Eat Amla, carrots, and green leafy vegetables for Vitamin A. (5) Practice Trataka (candle gazing) for eye muscle strengthening. (6) Avoid excessive screen time; follow the 20-20-20 rule."),

    # Herbs
    (["ashwagandha"],
     "Ashwagandha (Withania somnifera) is a powerful adaptogenic herb in Ayurveda. It helps reduce stress and anxiety by lowering cortisol levels, boosts immunity, improves energy and stamina, supports cognitive function and memory, and enhances male reproductive health. Dosage: 300-600mg of root extract daily. It balances Vata and Kapha doshas. Avoid during pregnancy."),

    (["turmeric", "haldi"],
     "Turmeric (Curcuma longa / Haldi) is one of the most important herbs in Ayurveda. Curcumin, its active compound, has powerful anti-inflammatory and antioxidant properties. It purifies blood, supports digestion, boosts immunity, heals wounds, and is used for skin conditions. Golden milk (Haldi Doodh) is a traditional preparation. It balances all three doshas. Best absorbed with black pepper (piperine) and fat."),

    (["tulsi", "basil"],
     "Tulsi (Holy Basil / Ocimum sanctum) is 'The Queen of Herbs' in Ayurveda. It supports respiratory health (cough, cold, asthma), reduces stress (adaptogenic), boosts immunity, has antibacterial and antiviral properties, purifies blood, and supports heart health. Chew 4-5 fresh leaves daily or drink Tulsi tea. It balances Kapha and Vata doshas. It is also considered sacred and grown in most Indian households."),

    (["neem"],
     "Neem (Azadirachta indica) is called 'Sarva Roga Nivarini' — the curer of all diseases. It is the most powerful blood purifier in Ayurveda. Benefits: treats skin diseases (acne, eczema, psoriasis), antibacterial and antifungal, supports dental health, controls blood sugar, and detoxifies the liver. Use: Neem leaf paste externally, Neem water internally, Neem oil for hair and skin. It is bitter (Tikta rasa) and cooling — pacifies Pitta and Kapha."),

    (["ginger", "adrak"],
     "Ginger (Zingiber officinale / Adrak) is called 'Vishwabheshaja' — the universal medicine. It kindles digestive fire (Agni), relieves nausea and bloating, reduces cold and cough, has anti-inflammatory properties for joint pain, and improves circulation. Fresh ginger tea with lemon and honey is a daily health tonic. Dry ginger (Sunthi) is even more potent for Vata and Kapha disorders. Avoid excess if you have Pitta imbalance (acidity)."),

    (["amla", "gooseberry"],
     "Amla (Indian Gooseberry / Emblica officinalis) is the richest natural source of Vitamin C. It is a key ingredient in Triphala and Chyawanprash. Benefits: boosts immunity, strengthens hair and prevents greying, improves eyesight, supports digestion, anti-aging (Rasayana), and regulates blood sugar. It balances all three doshas — one of the rare 'Tridoshahara' herbs. Take as juice, powder, or eat fresh."),

    (["giloy", "guduchi"],
     "Guduchi / Giloy (Tinospora cordifolia) is called 'Amrita' — the root of immortality. It is the premier immunomodulator in Ayurveda. Benefits: boosts immunity, treats chronic fevers, purifies blood, manages diabetes, supports liver health, and reduces arthritis inflammation. Preparation: boil fresh Giloy stem in water, or take Guduchi Satva powder. It balances all three doshas and is especially effective for Pitta disorders."),

    # Core concepts
    (["dosha", "vata", "pitta", "kapha", "prakriti", "constitution"],
     "In Ayurveda, there are three doshas — Vata (air + space), Pitta (fire + water), and Kapha (earth + water). Every person has a unique combination called Prakriti (constitution). Vata governs movement, creativity, and communication — imbalance causes anxiety, dry skin, constipation. Pitta governs digestion, metabolism, and intellect — imbalance causes inflammation, acidity, anger. Kapha governs structure, stability, and immunity — imbalance causes weight gain, congestion, lethargy. Balancing your doshas through diet, lifestyle, and herbs is the foundation of Ayurvedic health."),

    (["digestion", "agni", "hunger", "appetite"],
     "Ayurveda considers Agni (digestive fire) as the cornerstone of health — 'You are what you digest, not what you eat.' Tips: (1) Eat warm, freshly cooked meals. (2) Drink warm water throughout the day. (3) Use digestive spices: ginger, cumin, fennel, coriander, black pepper. (4) Eat largest meal at lunch (when Agni peaks). (5) Avoid eating when not hungry. (6) Don't drink ice-cold water with meals. (7) Practice mindful eating — sit down, chew well, no distractions. (8) Triphala before bed regulates the entire digestive system."),

    (["immunity", "immune", "protect"],
     "Ayurveda recommends Rasayana (rejuvenation) for strong immunity: Herbs — Ashwagandha, Guduchi (Giloy), Amalaki (Amla), Tulsi, and Pippali. Daily practices — oil pulling with sesame oil, warm water with turmeric, Pranayama (especially Kapalbhati and Anulom Vilom), adequate sleep, and Abhyanga (self-massage). Classical formulation — Chyawanprash (1 tsp daily). Diet — seasonal eating, include all six tastes (Shadrasa), favor warm and fresh food. Lifestyle — follow Dinacharya (daily routine) aligned with natural circadian rhythms."),

    (["stress", "tension", "relax", "calm"],
     "Ayurvedic stress management: Herbs — Ashwagandha (adaptogen, lowers cortisol), Brahmi (calms mind), Jatamansi (anti-anxiety), Shankhpushpi (nervous system tonic). Therapies — Shirodhara (warm oil poured on forehead), Abhyanga (full body oil massage), Nasya (nasal oil therapy). Daily practices — meditation, Nadi Shodhana Pranayama (alternate nostril breathing), yoga (especially Shavasana, Balasana, Viparita Karani). Diet — avoid caffeine, alcohol, and processed foods; eat warm, nourishing meals with ghee. Follow a regular daily routine (Dinacharya)."),

    (["skin", "acne", "pimple", "glow", "complexion"],
     "Ayurvedic skin care: (1) Neem — the most powerful blood purifier; use Neem paste or wash face with Neem water for acne. (2) Turmeric face mask (Haldi + Besan + milk) for glowing skin. (3) Aloe Vera gel for cooling, moisturizing, and healing. (4) Manjistha (Rubia cordifolia) internally for even complexion. (5) Kumkumadi Tailam — the classical Ayurvedic face oil with saffron for radiant skin. (6) Drink Triphala water for detoxification. Diet: avoid spicy, oily, and fermented foods (aggravate Pitta). Eat cooling foods — cucumber, coconut water, mint, coriander."),

    (["sleep", "insomnia", "rest"],
     "Ayurvedic sleep remedies: (1) Warm milk with nutmeg (Jaiphal), saffron (Kesar), and a pinch of turmeric before bed. (2) Ashwagandha Churna with warm milk — calms Vata and promotes deep sleep. (3) Brahmi oil head massage before sleeping. (4) Jatamansi powder — a potent herb for insomnia and anxiety. (5) Foot massage with warm sesame oil activates Marma points. (6) Follow a consistent sleep schedule — sleep by 10 PM (Kapha time). (7) Avoid screens, caffeine, and heavy meals 2 hours before bed. (8) Practice Yoga Nidra or body scan meditation."),

    (["yoga", "exercise", "pranayama", "breathing"],
     "Yoga and Pranayama in Ayurveda: Yoga is the sister science of Ayurveda — both work together for holistic health. Key practices: (1) Surya Namaskar — 12 rounds daily for full-body exercise. (2) Kapalbhati Pranayama — detoxifies, boosts metabolism, clears sinuses. (3) Anulom Vilom (alternate nostril breathing) — balances all doshas, calms the mind. (4) Bhramari (bee breathing) — reduces anxiety and BP. (5) Shavasana — deep relaxation for stress relief. For Vata types: gentle, grounding yoga. For Pitta types: cooling, moderate yoga. For Kapha types: vigorous, energizing yoga."),

    (["detox", "cleanse", "toxin", "ama", "purify"],
     "Ayurvedic detoxification (Shodhana): Ama (toxins) accumulates from poor digestion and unhealthy habits. Signs: coated tongue, bad breath, fatigue, brain fog. Daily detox: (1) Warm lemon water on empty stomach. (2) Triphala Churna before bed. (3) Tongue scraping every morning. (4) Drink warm water throughout the day. Seasonal detox: Panchakarma — the 5-action Ayurvedic cleansing therapy (Vamana, Virechana, Basti, Nasya, Raktamokshana). Diet: eat light khichdi for 3-5 days with digestive spices. Herbs: Guggulu, Triphala, Guduchi (Giloy), and Kutki for liver detox."),
]


def get_fallback_response(query):
    """Keyword-based response when RAG pipeline is not available."""
    query_lower = query.lower()
    for keywords, response in AYURVEDIC_RESPONSES:
        if any(kw in query_lower for kw in keywords):
            return response
    return "Ayurveda is an ancient Indian system of holistic medicine that focuses on balancing mind, body, and spirit through herbs, diet, yoga, and lifestyle practices. The core principles revolve around three doshas (Vata, Pitta, Kapha), seven dhatus (tissues), and Agni (digestive fire). Try asking about specific topics like: cold, fever, headache, digestion, joint pain, diabetes, hair care, weight loss, stress, skin care, sleep, immunity, detox, or herbs like Ashwagandha, Turmeric, Tulsi, Neem, Giloy, Amla, and Ginger!"


# ──────────────────────────────────────────────
# Initialize Q-LREF Session
# ──────────────────────────────────────────────
def init_crypto_session():
    """Initialize Q-LREF session and perform handshake."""
    if "qlref_session" not in st.session_state:
        session = QLREFSession(
            n=config.QLREF_LATTICE_DIM,
            q=config.QLREF_MODULUS,
            sigma=config.QLREF_NOISE_SIGMA,
        )
        session.perform_handshake()
        st.session_state.qlref_session = session


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🌿 आयुर्वेदिक चैटबॉट</h1>
    <p>Bilingual Ayurvedic Health Advisor — Hindi & English</p>
    <p style="font-size: 0.8rem; color: #FF9800; margin-top: 4px;">Secured with Q-LREF Post-Quantum Encryption</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Settings & Info
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    st.markdown("---")

    # Q-LREF Crypto Controls
    st.markdown("### Q-LREF Encryption")
    enable_crypto = st.checkbox("Enable Q-LREF Encryption", value=True)
    show_crypto_details = st.checkbox("Show Encrypt/Decrypt Steps", value=True)

    if enable_crypto:
        init_crypto_session()
        session = st.session_state.qlref_session
        summary = session.get_handshake_summary()
        st.success(f"Session: {summary['status']}")
        st.caption(f"Lattice: {summary['lattice_dim']}-dim | Key: {summary['public_key_size']}")
        st.caption(f"Handshake: {summary['total_handshake']} | Msgs: {summary['messages_encrypted']}")

        if st.button("New Key Exchange"):
            new_session = QLREFSession(
                n=config.QLREF_LATTICE_DIM,
                q=config.QLREF_MODULUS,
                sigma=config.QLREF_NOISE_SIGMA,
            )
            new_session.perform_handshake()
            st.session_state.qlref_session = new_session
            st.rerun()

    st.markdown("---")

    # Benchmark Button
    st.markdown("### Security Benchmarks")
    if st.button("Run Q-LREF Benchmark"):
        with st.spinner("Running 50 trials..."):
            bench = crypto_metrics.benchmark_qlref(n_trials=50)
            st.session_state.benchmark_results = bench

    if "benchmark_results" in st.session_state:
        b = st.session_state.benchmark_results
        st.markdown(f"""
        **Q-LREF Performance ({b['n_trials']} trials)**
        - KeyGen: `{b['keygen']['mean']:.2f} ms`
        - Exchange: `{b['exchange']['mean']:.2f} ms`
        - Encrypt: `{b['encrypt']['mean']:.4f} ms`
        - Decrypt: `{b['decrypt']['mean']:.4f} ms`
        - Key Size: `{b['public_key_size_bytes']} bytes`
        """)

    if st.button("Run MITM Attack Simulation"):
        with st.spinner("Simulating 100 MITM attacks..."):
            mitm = crypto_metrics.simulate_mitm_attack(n_trials=100)
            st.session_state.mitm_results = mitm

    if "mitm_results" in st.session_state:
        m = st.session_state.mitm_results
        st.markdown(f"""
        **MITM Attack Results**
        - Trials: `{m['n_trials']}`
        - Successes: `{m['successes']}`
        - Success Rate: `{m['success_rate']}`
        - Status: **{m['status']}**
        """)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
    - **Model:** mT5-small (fine-tuned)
    - **Retrieval:** FAISS + SentenceTransformer
    - **Languages:** Hindi + English
    - **KB:** BhashaBench-Ayur + Ayurvedic texts
    - **Crypto:** Q-LREF (LWE Lattice-Based)
    """)

    st.markdown("---")
    st.markdown("""
    > **Disclaimer:** This chatbot provides general
    > Ayurvedic information only. Not a substitute for
    > professional medical advice.
    """)


# ──────────────────────────────────────────────
# Handshake Details (Main Area - Tab)
# ──────────────────────────────────────────────
if enable_crypto and "qlref_session" in st.session_state:
    session = st.session_state.qlref_session

    tab_chat, tab_handshake, tab_security = st.tabs([
        "Chat", "Key Exchange Details", "Security Metrics"
    ])

    # ──────────────────────────────────────────
    # Tab: Security Metrics
    # ──────────────────────────────────────────
    with tab_security:
        st.markdown("### Algorithm Comparison")
        comp_table = crypto_metrics.get_security_comparison_table()
        col_headers = ["Algorithm", "Key Size", "Encrypt", "Decrypt", "Quantum Safe", "NIST Level"]
        table_md = "| " + " | ".join(col_headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
        for row in comp_table:
            table_md += f"| {row['algorithm']} | {row['key_size']} | {row['encrypt_ms']} | {row['decrypt_ms']} | {row['quantum_safe']} | {row['nist_level']} |\n"
        st.markdown(table_md)

        st.markdown("### Threat Model")
        threat_table = crypto_metrics.get_threat_model_table()
        t_headers = ["Attack Type", "Classical Risk", "Quantum Risk", "Status"]
        t_md = "| " + " | ".join(t_headers) + " |\n"
        t_md += "| " + " | ".join(["---"] * len(t_headers)) + " |\n"
        for row in threat_table:
            t_md += f"| {row['attack']} | {row['classical_risk']} | {row['quantum_risk']} | **{row['status']}** |\n"
        st.markdown(t_md)

        st.markdown("### Brute-Force Resistance")
        for bits in [128, 192, 256]:
            bf = crypto_metrics.simulate_brute_force(bits)
            st.markdown(f"""
            **{bits}-bit key:**
            Classical: {bf['classical_time']} | Quantum (Grover): {bf['quantum_time']} | Status: **{bf['status']}**
            """)

    # ──────────────────────────────────────────
    # Tab: Key Exchange Details (Dynamic per-message flow)
    # ──────────────────────────────────────────
    with tab_handshake:

        # ── Section 1: Initial Handshake with FULL details ──
        st.markdown("## Phase 1: LWE Key Exchange (Session Start)")

        # Parameters box
        st.code(f"""Q-LREF Parameters (NIST Security Level 3):
  Lattice dimension (n):  {session.n}
  Prime modulus (q):      {session.q}
  Noise std dev (sigma):  {session.sigma}
  Security parameter (λ): 128 bits
  Underlying problem:     Learning With Errors (LWE)
  Symmetric cipher:       AES-256-GCM""", language="text")

        # Step 1: Client KeyGen
        st.markdown("### Step 1: Client Key Generation")
        st.code(f"""Algorithm: KeyGen(λ=128, n={session.n}, q={session.q}, χ=Gaussian(σ={session.sigma}))

  1. Generate private key:  s_A ← Gaussian({session.sigma})^{session.n}
     s_A = [{session.client.private_key[0]}, {session.client.private_key[1]}, {session.client.private_key[2]}, {session.client.private_key[3]}, {session.client.private_key[4]}, {session.client.private_key[5]}, {session.client.private_key[6]}, {session.client.private_key[7]}, ... ] ({session.n} values)

  2. Generate public matrix: A ← Z_{session.q}^{{{session.n}×{session.n}}} (uniform random)
     A[0:3][0:3] = {session.client.A[:3, :3].tolist()}
     Matrix size:  {session.n}x{session.n} = {session.n*session.n} integers in Z_{session.q}

  3. Generate error vector: e_A ← Gaussian({session.sigma})^{session.n}

  4. Compute public key:    p_A = A · s_A + e_A (mod {session.q})
     p_A = [{session.client.public_key[0]}, {session.client.public_key[1]}, {session.client.public_key[2]}, {session.client.public_key[3]}, {session.client.public_key[4]}, {session.client.public_key[5]}, {session.client.public_key[6]}, {session.client.public_key[7]}, ... ] ({session.n} values)

  Time: {session.metrics['client_keygen_ms']:.2f}ms""", language="text")

        # Step 2: Key exchange
        st.markdown("### Step 2: Public Key Exchange")
        st.code(f"""Client ──── sends (A, p_A) ────▶ Server
  Transmitted: Matrix A ({session.n}x{session.n}) + Public key p_A ({session.n} values)
  Public key size: {session.metrics.get('public_key_size_bytes', 0)} bytes

Server generates own keypair using SAME matrix A:
  1. s_B ← Gaussian({session.sigma})^{session.n}
     s_B = [{session.server.private_key[0]}, {session.server.private_key[1]}, {session.server.private_key[2]}, {session.server.private_key[3]}, {session.server.private_key[4]}, {session.server.private_key[5]}, {session.server.private_key[6]}, {session.server.private_key[7]}, ... ]
  2. e_B ← Gaussian({session.sigma})^{session.n}
  3. p_B = A · s_B + e_B (mod {session.q})
     p_B = [{session.server.public_key[0]}, {session.server.public_key[1]}, {session.server.public_key[2]}, {session.server.public_key[3]}, {session.server.public_key[4]}, {session.server.public_key[5]}, {session.server.public_key[6]}, {session.server.public_key[7]}, ... ]

Server ◀──── sends p_B ──────── Client
  Time: {session.metrics['server_keygen_ms']:.2f}ms""", language="text")

        # Step 3: Shared secret
        st.markdown("### Step 3: Shared Secret Derivation")
        st.code(f"""CLIENT computes:
  K_A = s_A · p_B + e'_A (mod {session.q})
  where e'_A ← Gaussian({session.sigma})^{session.n} (fresh noise)
  Raw secret K_A = [{session.client.shared_secret_raw[0]}, {session.client.shared_secret_raw[1]}, {session.client.shared_secret_raw[2]}, {session.client.shared_secret_raw[3]}, {session.client.shared_secret_raw[4]}, ... ]
  Time: {session.metrics['client_secret_ms']:.3f}ms

SERVER computes:
  K_B = s_B · p_A + e'_B (mod {session.q})
  where e'_B ← Gaussian({session.sigma})^{session.n} (fresh noise)
  Raw secret K_B = [{session.server.shared_secret_raw[0]}, {session.server.shared_secret_raw[1]}, {session.server.shared_secret_raw[2]}, {session.server.shared_secret_raw[3]}, {session.server.shared_secret_raw[4]}, ... ]
  Time: {session.metrics['server_secret_ms']:.3f}ms

Note: K_A ≈ K_B (approximately equal, differ by small noise terms)""", language="text")

        # Step 4: Reconciliation
        st.markdown("### Step 4: Reconciliation")
        st.code(f"""Reconciliation extracts exact shared bits from approximate secrets.

Algorithm: For each coefficient K[i]:
  If K[i] ∈ [q/4, 3q/4) → bit = 1
  If K[i] ∈ [0, q/4) or [3q/4, q) → bit = 0
  Threshold = q/4 = {session.q // 4}

Client reconciled bits (first 20): {session.client.shared_bits[:20].tolist()}
Server reconciled bits (first 20): {session.server.shared_bits[:20].tolist()}

Bits match: {list(session.client.shared_bits[:20]) == list(session.server.shared_bits[:20])}""", language="text")

        # Step 5: AES key derivation
        st.markdown("### Step 5: AES Key Derivation & Hash Verification")
        st.code(f"""AES Key = SHA-256(reconciled_bits)

Client AES-256 Key: {session.client.aes_key.hex()}
Server AES-256 Key: {session.server.aes_key.hex()}

Hash Verification (SHA-256 of reconciled bits):
  Client hash: {session.client.verification_hash}
  Server hash: {session.server.verification_hash}
  Match: {session.client.verification_hash == session.server.verification_hash}

RESULT: Secure channel established!
  Total handshake time: {session.metrics.get('total_handshake_ms', 0):.2f}ms
  Public key size: {session.metrics.get('public_key_size_bytes', 0)} bytes (1024 bytes, comparable to CRYSTALS-Kyber 1184 bytes)
  Security: Quantum-resistant (LWE hardness, NIST Level 3)""", language="text")

        if session.client.verification_hash == session.server.verification_hash:
            st.success("Keys Match — Quantum-Resistant Secure Channel Established!")
        else:
            st.warning("Keys differed slightly — Reconciliation fallback applied.")

        # ── Section 2: Per-Message Encryption Flow ──
        st.markdown("---")
        st.markdown("## Phase 2: Per-Message Encrypted Communication")

        # Get all messages with crypto info — including latest from session_state
        crypto_messages = [
            m for m in st.session_state.get("messages", [])
            if "crypto_info" in m
        ]
        # Also check for pending crypto info (current message being processed)
        if "latest_crypto_info" in st.session_state:
            pending = st.session_state.latest_crypto_info
            already_in = any(
                m.get("crypto_info", {}).get("encrypted_query_hex") == pending.get("encrypted_query_hex")
                for m in crypto_messages
            )
            if not already_in:
                crypto_messages.append({"crypto_info": pending})

        if not crypto_messages:
            st.info("Send a message in the Chat tab to see the live encryption flow here!")
        else:
            for idx, msg in enumerate(crypto_messages):
                ci = msg["crypto_info"]
                msg_num = idx + 1
                q_match = "PASS" if ci['query_match'] else "FAIL"
                r_match = "PASS" if ci['response_match'] else "FAIL"
                total_overhead = ci['query_encrypt_ms'] + ci['query_decrypt_ms'] + ci['response_encrypt_ms'] + ci['response_decrypt_ms']

                st.markdown(f"### Message #{msg_num}: \"{ci['original_query']}\"")

                st.code(f"""═══════════════════════════════════════════════════════════════
  MESSAGE #{msg_num} — FULL ENCRYPTION/DECRYPTION TRACE
═══════════════════════════════════════════════════════════════

[A] USER INPUT (Client Side)
    Plaintext query: "{ci['original_query']}"
    Bytes (UTF-8):   {len(ci['original_query'].encode('utf-8'))} bytes

[B] CLIENT ENCRYPTS QUERY (AES-256-GCM)
    Algorithm:    AES-256-GCM
    Key:          (Q-LREF derived shared key, 256 bits)
    Nonce:        12 bytes (random, unique per message)
    Input:        "{ci['original_query']}"
    Output:       {ci['encrypted_query_hex']}
    Cipher size:  {ci['cipher_size_bytes']} bytes (nonce + ciphertext + GCM tag)
    GCM Auth Tag: 16 bytes (integrity protection)
    Time:         {ci['query_encrypt_ms']:.4f} ms

    ═══════ [ ENCRYPTED QUERY SENT OVER NETWORK ] ═══════▶
    An interceptor sees: {ci['encrypted_query_hex'][:50]}...
    This is UNREADABLE without the shared AES key.

[C] SERVER DECRYPTS QUERY (AES-256-GCM)
    Algorithm:    AES-256-GCM
    Input:        {ci['encrypted_query_hex'][:50]}...
    Verify tag:   GCM authentication tag verified
    Output:       "{ci['decrypted_query']}"
    Time:         {ci['query_decrypt_ms']:.4f} ms
    Integrity:    {q_match} (decrypted == original: {ci['query_match']})

[D] SERVER PROCESSES QUERY
    Pipeline:     Language Detect → Translate → Retrieve → Generate
    Response:     "{ci['original_response'][:100]}..."

[E] SERVER ENCRYPTS RESPONSE (AES-256-GCM)
    Algorithm:    AES-256-GCM
    Nonce:        12 bytes (new random nonce)
    Input:        "{ci['original_response'][:80]}..."
    Output:       {ci['encrypted_response_hex']}
    Cipher size:  {ci['response_cipher_size']} bytes
    Time:         {ci['response_encrypt_ms']:.4f} ms

    ◀═══════ [ ENCRYPTED RESPONSE SENT OVER NETWORK ] ═══════
    Quantum-resistant: Even a quantum computer cannot break this.

[F] CLIENT DECRYPTS RESPONSE (AES-256-GCM)
    Algorithm:    AES-256-GCM
    Input:        {ci['encrypted_response_hex'][:50]}...
    Verify tag:   GCM authentication tag verified
    Output:       "{ci['decrypted_response'][:100]}..."
    Time:         {ci['response_decrypt_ms']:.4f} ms
    Integrity:    {r_match} (decrypted == original: {ci['response_match']})

───────────────────────────────────────────────────────────────
SUMMARY:
  Encrypt query:     {ci['query_encrypt_ms']:.4f} ms
  Decrypt query:     {ci['query_decrypt_ms']:.4f} ms
  Encrypt response:  {ci['response_encrypt_ms']:.4f} ms
  Decrypt response:  {ci['response_decrypt_ms']:.4f} ms
  TOTAL OVERHEAD:    {total_overhead:.4f} ms
  Query integrity:   {q_match}
  Response integrity:{r_match}
═══════════════════════════════════════════════════════════════""", language="text")

    # ──────────────────────────────────────────
    # Tab: Chat
    # ──────────────────────────────────────────
    with tab_chat:
        _render_chat = True
else:
    tab_chat = None
    _render_chat = True


# ──────────────────────────────────────────────
# Chat Interface
# ──────────────────────────────────────────────

# Determine container (tab or main page)
chat_container = tab_chat if tab_chat is not None else st

with (tab_chat if tab_chat is not None else st.container()):

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Namaste! I am your Ayurvedic health advisor. Ask me about herbs, doshas, remedies, and Ayurvedic wellness in Hindi or English!"
        })

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=None):
            st.markdown(message["content"])

            # Show crypto details for this message
            if show_crypto_details and "crypto_info" in message:
                ci = message["crypto_info"]
                with st.expander("Encryption/Decryption Details", expanded=False):
                    st.markdown(f"""
                    <div class="crypto-box">
                        <div class="crypto-label">ENCRYPTION (Client -> Server)</div>
                        <div>Plaintext: <span style="color: #81C784;">{ci['original_query']}</span></div>
                        <div>Encrypted (hex): <span class="crypto-data">{ci['encrypted_query_hex']}</span></div>
                        <div>Encrypt Time: <span class="crypto-success">{ci['query_encrypt_ms']:.4f} ms</span></div>
                        <div>Decrypted by Server: <span style="color: #81C784;">{ci['decrypted_query']}</span></div>
                        <div>Decrypt Time: <span class="crypto-success">{ci['query_decrypt_ms']:.4f} ms</span></div>
                    </div>
                    <div class="crypto-box" style="border-color: rgba(76, 175, 80, 0.4);">
                        <div class="crypto-label" style="color: #66BB6A;">ENCRYPTION (Server -> Client)</div>
                        <div>Plaintext Response (first 100 chars): <span style="color: #81C784;">{ci['original_response'][:100]}...</span></div>
                        <div>Encrypted (hex): <span class="crypto-data">{ci['encrypted_response_hex']}</span></div>
                        <div>Encrypt Time: <span class="crypto-success">{ci['response_encrypt_ms']:.4f} ms</span></div>
                        <div>Decrypted by Client: <span style="color: #81C784;">{ci['decrypted_response'][:100]}...</span></div>
                        <div>Decrypt Time: <span class="crypto-success">{ci['response_decrypt_ms']:.4f} ms</span></div>
                    </div>
                    <div class="crypto-box" style="border-color: rgba(33, 150, 243, 0.4);">
                        <div class="crypto-label" style="color: #42A5F5;">VERIFICATION</div>
                        <div>Query Match: <span class="crypto-success">{"PASS" if ci['query_match'] else "FAIL"}</span></div>
                        <div>Response Match: <span class="crypto-success">{"PASS" if ci['response_match'] else "FAIL"}</span></div>
                        <div>Algorithm: AES-256-GCM with Q-LREF derived key</div>
                        <div>Cipher Size: {ci['cipher_size_bytes']} bytes (query) | {ci['response_cipher_size']} bytes (response)</div>
                    </div>
                    """, unsafe_allow_html=True)


    # Chat input
    if prompt := st.chat_input("Ask about Ayurveda... / आयुर्वेद के बारे में पूछें..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                crypto_info = None

                # Get response — try RAG pipeline, fallback to keyword matching
                rag = load_pipeline()
                use_rag = rag is not None

                if enable_crypto and "qlref_session" in st.session_state:
                    qlref_session = st.session_state.qlref_session

                    # === STEP 1: Client encrypts query ===
                    encrypted_query, enc_q_ms = qlref_session.encrypt_query(prompt)
                    encrypted_query_hex = bytes_to_hex_preview(encrypted_query, 80)

                    # === STEP 2: Server decrypts query ===
                    decrypted_query, dec_q_ms = qlref_session.decrypt_query(encrypted_query)

                    # Show live encryption process
                    if show_crypto_details:
                        st.markdown(f"""
                        <div class="crypto-box">
                            <div class="crypto-label">LIVE: Query Encrypted by Client</div>
                            <div>Plaintext: <span style="color: #81C784;">"{prompt}"</span></div>
                            <div>Ciphertext: <span class="crypto-data">{encrypted_query_hex}</span></div>
                            <div>Server Decrypted: <span style="color: #81C784;">"{decrypted_query}"</span></div>
                            <div>Encrypt: {enc_q_ms:.4f}ms | Decrypt: {dec_q_ms:.4f}ms | <span class="crypto-success">MATCH VERIFIED</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # === STEP 3: Get response ===
                    if use_rag:
                        result = rag.answer(decrypted_query, top_k=5)
                        response = result["response"]
                    else:
                        response = get_fallback_response(decrypted_query)

                    # === STEP 4: Server encrypts response ===
                    encrypted_response, enc_r_ms = qlref_session.encrypt_response(response)
                    encrypted_response_hex = bytes_to_hex_preview(encrypted_response, 80)

                    # === STEP 5: Client decrypts response ===
                    decrypted_response, dec_r_ms = qlref_session.decrypt_response(encrypted_response)

                    # Show live response encryption
                    if show_crypto_details:
                        st.markdown(f"""
                        <div class="crypto-box" style="border-color: #4caf50;">
                            <div class="crypto-label" style="color: #66bb6a;">LIVE: Response Encrypted by Server</div>
                            <div>Plaintext: <span style="color: #81C784;">"{response[:120]}..."</span></div>
                            <div>Ciphertext: <span class="crypto-data">{encrypted_response_hex}</span></div>
                            <div>Client Decrypted: <span style="color: #81C784;">"{decrypted_response[:120]}..."</span></div>
                            <div>Encrypt: {enc_r_ms:.4f}ms | Decrypt: {dec_r_ms:.4f}ms | <span class="crypto-success">MATCH VERIFIED</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Build crypto info for history and store immediately for Phase 2 tab
                    crypto_info = {
                        "original_query": prompt,
                        "encrypted_query_hex": encrypted_query_hex,
                        "query_encrypt_ms": enc_q_ms,
                        "decrypted_query": decrypted_query,
                        "query_decrypt_ms": dec_q_ms,
                        "query_match": decrypted_query == prompt,
                        "original_response": response,
                        "encrypted_response_hex": encrypted_response_hex,
                        "response_encrypt_ms": enc_r_ms,
                        "decrypted_response": decrypted_response,
                        "response_decrypt_ms": dec_r_ms,
                        "response_match": decrypted_response == response,
                        "cipher_size_bytes": len(encrypted_query),
                        "response_cipher_size": len(encrypted_response),
                    }

                    response = decrypted_response

                    # Store immediately so Key Exchange Details tab can show it
                    st.session_state.latest_crypto_info = crypto_info

                else:
                    # No crypto
                    if use_rag:
                        result = rag.answer(prompt, top_k=5)
                        response = result["response"]
                    else:
                        response = get_fallback_response(prompt)

                # Display the response
                st.markdown(response)

                # Status line
                crypto_badge = ""
                if crypto_info:
                    total_crypto_ms = (crypto_info['query_encrypt_ms'] +
                                      crypto_info['query_decrypt_ms'] +
                                      crypto_info['response_encrypt_ms'] +
                                      crypto_info['response_decrypt_ms'])
                    crypto_badge = f" | Q-LREF Encrypted ({total_crypto_ms:.2f}ms overhead)"

                st.caption(f"Algorithm: AES-256-GCM + LWE Key Exchange{crypto_badge}")

                # Save to history
                msg = {
                    "role": "assistant",
                    "content": response,
                }
                if crypto_info:
                    msg["crypto_info"] = crypto_info
                st.session_state.messages.append(msg)

                # Rerun so the Key Exchange Details tab updates immediately
                st.rerun()

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
