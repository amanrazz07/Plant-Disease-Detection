"""
Agronomic knowledge base providing diagnosis, treatment, and prevention
remedies for plant diseases detected in the PlantVillage dataset.
"""

REMEDIES_DB = {
    # Apple
    "Apple___Apple_scab": {
        "cause": "Fungus (Venturia inaequalis) thriving in wet, cool spring weather.",
        "treatment": "Apply fungicides containing captan, mancozeb, or copper soap. Prune infected branches to improve air circulation.",
        "prevention": "Rake and destroy fallen leaves in autumn. Plant scab-resistant cultivars and avoid overhead watering."
    },
    "Apple___Black_rot": {
        "cause": "Fungus (Botryosphaeria obtusa) entering via wounds or insect punctures.",
        "treatment": "Prune out dead wood, cankers, and mummified fruit. Apply captan or sulfur-based spray during bloom.",
        "prevention": "Maintain tree vigor with balanced fertilizer and keep canopies well-ventilated through winter pruning."
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Fungus (Gymnosporangium juniperi-virginianae) alternating between junipers and apple trees.",
        "treatment": "Apply myclobutanil or sulfur fungicide at blossom bud break.",
        "prevention": "Remove nearby eastern red cedar trees or galls within several hundred yards of the orchard."
    },

    # Corn / Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Fungus (Cercospora zeae-maydis) spread by wind and rain splash in humid conditions.",
        "treatment": "Foliar fungicides (strobilurins or triazoles) applied at tasseling stage if disease severity is high.",
        "prevention": "Rotate crops with non-grasses like soybeans. Utilize tillage to decompose infected crop residue."
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Fungus (Puccinia sorghi) causing reddish-brown pustules on both leaf surfaces.",
        "treatment": "Usually not required unless infection occurs early. Triazole fungicides can halt severe outbreaks.",
        "prevention": "Plant resistant hybrid corn varieties and ensure early planting to avoid peak spore migration."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Fungus (Exserohilum turcicum) causing cigar-shaped grayish-green lesions.",
        "treatment": "Apply strobilurin or triazole fungicides at or before tasseling when lesions appear on upper leaves.",
        "prevention": "Practice 2-year crop rotation and plant resistant hybrids with Ht gene resistance."
    },

    # Grape
    "Grape___Black_rot": {
        "cause": "Fungus (Guignardia bidwellii) attacking young leaves, shoots, and berries.",
        "treatment": "Apply myclobutanil, mancozeb, or kresoxim-methyl starting when new shoots are 1-3 inches long.",
        "prevention": "Remove all mummified berries from vines and ground. Keep canopy open to direct sunlight."
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Complex of fungal pathogens (Phaeomoniella, Phaeoacremonium) causing wood decay.",
        "treatment": "No chemical cure exists for infected vines. Cut out severely infected cordons and seal pruning wounds.",
        "prevention": "Disinfect pruning shears between vines. Avoid pruning during wet weather."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Fungus (Phaeoisariopsis vitis) leading to dark angular leaf lesions.",
        "treatment": "Spray with copper hydroxide or broad-spectrum fungicides.",
        "prevention": "Ensure good canopy airflow, rake fallen leaves, and manage weed growth under vines."
    },

    # Potato
    "Potato___Early_blight": {
        "cause": "Fungus (Alternaria solani) producing concentric 'bullseye' dark rings on older foliage.",
        "treatment": "Apply chlorothalonil, azoxystrobin, or copper fungicide as soon as symptoms emerge.",
        "prevention": "Avoid overhead irrigation. Ensure adequate nitrogen and potassium fertility; rotate with non-solanaceous crops."
    },
    "Potato___Late_blight": {
        "cause": "Oomycete (Phytophthora infestans) — highly destructive pathogen responsible for the Irish Potato Famine.",
        "treatment": "Immediate application of systemic fungicides like metalaxyl, cymoxanil, or copper-based sprays. Remove infected plants.",
        "prevention": "Plant certified disease-free seed tubers. Eliminate cull piles, destroy volunteer potato plants, and ensure dry foliage."
    },

    # Tomato
    "Tomato___Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas spp.) spread by rain splash and contaminated seed.",
        "treatment": "Apply copper bactericide combined with mancozeb to slow bacterial spread.",
        "prevention": "Use pathogen-free seed, avoid working in wet fields, and practice 3-year crop rotation away from peppers/tomatoes."
    },
    "Tomato___Early_blight": {
        "cause": "Fungus (Alternaria linariae/solani) attacking bottom leaves with dark concentric rings.",
        "treatment": "Remove lower infected foliage. Spray with copper, chlorothalonil, or bio-fungicides (Bacillus subtilis).",
        "prevention": "Mulch soil around base to prevent soil splash. Stake plants and water strictly at soil level."
    },
    "Tomato___Late_blight": {
        "cause": "Water mold (Phytophthora infestans) causing rapid dark, water-soaked rot on leaves and fruit.",
        "treatment": "Apply copper or chlorothalonil immediately. Severely infected plants must be bagged and destroyed.",
        "prevention": "Ensure wide plant spacing for airflow. Do not plant near potatoes; monitor regional blight forecasts."
    },
    "Tomato___Leaf_Mold": {
        "cause": "Fungus (Passalora fulva) flourishing in high relative humidity (>85%), common in greenhouses.",
        "treatment": "Improve greenhouse ventilation with fans. Apply copper or bio-fungicide sprays.",
        "prevention": "Maintain relative humidity below 80%. Water early in the day and prune sucker branches."
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Fungus (Septoria lycopersici) causing small circular spots with dark brown margins and gray centers.",
        "treatment": "Apply copper soap, mancozeb, or sulfur. Strip away infected lower leaves.",
        "prevention": "Rotate crops every 2-3 years, avoid sprinkler irrigation, and clean all garden stakes with 10% bleach."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": "Arachnid pest (Tetranychus urticae) sucking plant sap in hot, dry conditions.",
        "treatment": "Spray insecticidal soap, neem oil, or horticultural oils on the undersides of leaves. Introduce predatory mites.",
        "prevention": "Keep plants adequately watered to reduce drought stress. Hose down foliage with strong water jets."
    },
    "Tomato___Target_Spot": {
        "cause": "Fungus (Corynespora cassiicola) causing brown lesions with pale concentric rings on leaves and fruit.",
        "treatment": "Apply chlorothalonil, azoxystrobin, or copper fungicide early in the infection cycle.",
        "prevention": "Avoid excess nitrogen fertilizer, ensure prompt removal of crop debris, and provide good air circulation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Begomovirus transmitted by silverleaf whiteflies (Bemisia tabaci).",
        "treatment": "No cure exists for viral infections. Pull and safely dispose of infected plants to protect healthy ones.",
        "prevention": "Control whitefly vectors using yellow sticky traps, insecticidal soap, or protective row covers."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "Stable Tobamovirus transmitted mechanically via hands, tools, and tobacco products.",
        "treatment": "Infected plants cannot be cured. Discard immediately to prevent spread.",
        "prevention": "Wash hands with soap and water after handling tobacco. Disinfect pruners with 20% nonfat dry milk or bleach."
    },

    # Pepper
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas campestris pv. vesicatoria).",
        "treatment": "Spray fixed copper mixed with mancozeb. Prune infected stems during dry afternoons.",
        "prevention": "Use hot-water treated seeds, practice drip irrigation, and avoid overhead sprinkler systems."
    },

    # Strawberry
    "Strawberry___Leaf_scorch": {
        "cause": "Fungus (Diplocarpon earlianum) causing purple/red blotches that turn brown.",
        "treatment": "Apply captan or copper fungicide before blooms open. Remove affected leaves after harvest.",
        "prevention": "Renovate strawberry beds after harvest by mowing and clearing older foliage. Ensure well-drained soil."
    },

    # Cherry & Peach
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Fungus (Podosphaera clandestina) causing white powdery fungal patches on new growth.",
        "treatment": "Apply sulfur or potassium bicarbonate sprays during early shoot development.",
        "prevention": "Prune to open the canopy to sunlight; avoid excessive late-summer nitrogen applications."
    },
    "Peach___Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas arboricola pv. pruni) causing shot-hole lesions on foliage.",
        "treatment": "Apply dormant copper sprays in late fall and early spring before bud break.",
        "prevention": "Plant resistant cultivars, avoid sandy windblown sites, and maintain balanced soil fertility."
    }
}


def get_remedy(class_name: str) -> dict:
    """Return cause, treatment, and prevention info for a given disease class."""
    if class_name.endswith("___healthy"):
        parts = class_name.split("___")
        plant = parts[0].replace("_", " ")
        return {
            "cause": "None — The plant exhibits vigorous foliage with no active pathogenic lesions.",
            "treatment": "No treatment required. Maintain current care, balanced watering, and nutrient schedule.",
            "prevention": f"Continue standard preventative maintenance: ensure 6-8 hours of sunlight, avoid waterlogging, and inspect leaves periodically."
        }

    if class_name in REMEDIES_DB:
        return REMEDIES_DB[class_name]

    # General fallback for any unlisted disease
    return {
        "cause": "Pathogenic microbial or fungal infection affecting leaf tissue.",
        "treatment": "Isolate the plant, prune severely infected leaves with sanitized tools, and apply an organic copper-based fungicide or neem spray.",
        "prevention": "Ensure good air circulation, avoid wetting leaves when watering, and rotate crops annually."
    }
