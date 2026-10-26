"""Pakistani name lists and city/alias data for the synthetic register generator.

These lists are written directly into the code so no external services are needed.
All names are common Pakistani names; none are real patient records.
"""

# ── Given names ──────────────────────────────────────────────────────────────
# Male given names with frequency weights (higher = more common)
MALE_FIRST_NAMES: list[tuple[str, int]] = [
    ("Muhammad", 200), ("Ali", 120), ("Ahmed", 110), ("Hassan", 90),
    ("Usman", 85),  ("Omar", 80),  ("Bilal", 75),  ("Hamza", 70),
    ("Ibrahim", 65), ("Zain", 60),  ("Asad", 55),  ("Saad", 55),
    ("Kashif", 50),  ("Fahad", 50),  ("Tariq", 48),  ("Irfan", 45),
    ("Adnan", 44),  ("Faisal", 44),  ("Kamran", 42), ("Rizwan", 40),
    ("Nadeem", 38),  ("Junaid", 37),  ("Wasim", 36),  ("Imran", 35),
    ("Azhar", 33),  ("Naveed", 32),  ("Farhan", 31),  ("Shahid", 30),
    ("Asif", 28),   ("Shoaib", 28),  ("Talha", 27),  ("Umer", 26),
    ("Waqas", 25),  ("Suliman", 24), ("Atif", 22),   ("Raza", 21),
    ("Jawad", 20),  ("Sajid", 20),   ("Tanveer", 18), ("Khalid", 18),
    ("Babar", 17),  ("Zubair", 17),  ("Shakeel", 16), ("Abid", 15),
    ("Zahid", 15),  ("Nasir", 14),   ("Mudassar", 13), ("Saeed", 12),
    ("Tahir", 12),  ("Yasir", 11),
]

FEMALE_FIRST_NAMES: list[tuple[str, int]] = [
    ("Fatima", 160), ("Ayesha", 140), ("Zainab", 110), ("Maryam", 100),
    ("Sara", 90),   ("Amna", 80),   ("Sana", 78),   ("Hina", 72),
    ("Nadia", 68),  ("Rabia", 65),  ("Asma", 60),   ("Saira", 58),
    ("Uzma", 55),   ("Noor", 54),   ("Shazia", 50),  ("Iram", 48),
    ("Bushra", 45),  ("Iqra", 44),   ("Kiran", 42),   ("Naima", 40),
    ("Samina", 38),  ("Rukhsar", 36), ("Mehwish", 35), ("Sadia", 33),
    ("Nimra", 32),  ("Afshan", 30),  ("Lubna", 28),   ("Farida", 27),
    ("Parveen", 25), ("Sobia", 24),  ("Shaista", 22), ("Razia", 20),
    ("Shabana", 19), ("Tahira", 18), ("Nasreen", 17), ("Ghazala", 16),
    ("Mehnaz", 15),  ("Shahida", 14), ("Rubia", 13),  ("Anjum", 12),
    ("Rehana", 11),  ("Farzana", 10), ("Nabeela", 10), ("Sadaf", 9),
    ("Anam", 8),    ("Humaira", 8),  ("Qurat", 7),   ("Mariam", 7),
    ("Tayyaba", 6),  ("Javeria", 6),
]

# ── Family names ─────────────────────────────────────────────────────────────
FAMILY_NAMES: list[tuple[str, int]] = [
    ("Khan", 300),  ("Ahmed", 250), ("Ali", 200),  ("Hassan", 150),
    ("Malik", 140), ("Sheikh", 130), ("Chaudhry", 120), ("Akhtar", 110),
    ("Butt", 100),  ("Qureshi", 95), ("Siddiqui", 90), ("Baig", 85),
    ("Mirza", 80),  ("Shah", 78),   ("Raza", 75),  ("Hussain", 72),
    ("Rana", 70),   ("Mughal", 68), ("Abbasi", 65), ("Nawaz", 62),
    ("Javed", 60),  ("Riaz", 58),   ("Zaman", 55),  ("Naqvi", 52),
    ("Rehman", 50), ("Asghar", 48), ("Ansari", 46), ("Farooq", 44),
    ("Bhatti", 42), ("Gillani", 40), ("Hashmi", 38), ("Tarar", 36),
    ("Warraich", 35), ("Gondal", 33), ("Dogar", 31), ("Cheema", 30),
    ("Khawaja", 28), ("Bokhari", 27), ("Zaidi", 25), ("Toor", 24),
    ("Chattha", 22), ("Gujjar", 21), ("Sandhu", 20), ("Dhillon", 18),
    ("Niazi", 17),  ("Baloch", 16), ("Mengal", 14), ("Bugti", 12),
    ("Marri", 11),  ("Leghari", 10),
]

# ── Spelling-variant maps (canonical -> list of common corrupted forms) ───────
# Used when creating duplicate copies of a record.
NAME_SPELLING_VARIANTS: dict[str, list[str]] = {
    # First-name variants
    "Muhammad": ["Mohammad", "Mohd", "M.", "Mohammed", "Muhammed", "Muhamad"],
    "Ahmed":    ["Ahmad", "Ahmd", "Ahamed"],
    "Ali":      ["Alii", "Aly"],
    "Hassan":   ["Hasan", "Husan"],
    "Usman":    ["Uthman", "Osman", "Usmann"],
    "Omar":     ["Umar", "Omer"],
    "Ibrahim":  ["Ibraheem", "Ibrahem"],
    "Ayesha":   ["Aisha", "Aiesha", "Aysha"],
    "Fatima":   ["Fateema", "Fatema", "Fatimah"],
    "Zainab":   ["Zainub", "Zienab"],
    "Maryam":   ["Mariam", "Maryum"],
    "Hamza":    ["Hamzah", "Hamzaa"],
    # Family-name variants
    "Khan":     ["Khaan", "Kahn"],
    "Ahmed":    ["Ahmad", "Ahmd"],
    "Chaudhry": ["Chaudhary", "Chaudhri", "Ch."],
    "Siddiqui": ["Siddique", "Siddiqi"],
    "Hussain":  ["Husain", "Husein"],
    "Rehman":   ["Rahman", "Rehmann"],
    "Qureshi":  ["Qurashi", "Quraishi", "Qreshi"],
}

HONORIFICS: list[str] = [
    "Mr.", "Haji", "Hafiz", "Dr.", "Engr.", "Mian", "Ch.",
    "Syed", "Agha",
]

# ── City lists and aliases ────────────────────────────────────────────────────
# Each tuple: (canonical_name, alias_forms, frequency_weight)
CITY_DATA: list[tuple[str, list[str], int]] = [
    ("Lahore",      ["LHR", "lahore", "Lhr", "LAHORE", "lhr"],    250),
    ("Karachi",     ["KHI", "karachi", "Khi", "KARACHI"],          220),
    ("Islamabad",   ["ISB", "islamabad", "Isb", "I'abad"],         180),
    ("Rawalpindi",  ["RWP", "rawalpindi", "Rwp", "Pindi"],         150),
    ("Faisalabad",  ["FSD", "faisalabad", "Fsd", "Lyallpur"],      130),
    ("Multan",      ["MLT", "multan", "Mlt", "MULTAN"],            110),
    ("Peshawar",    ["PEW", "peshawar", "Pew", "PESHAWAR"],         90),
    ("Quetta",      ["QTA", "quetta", "Qta", "QUETTA"],             75),
    ("Gujranwala",  ["GRW", "gujranwala", "Grw"],                   70),
    ("Sialkot",     ["SKT", "sialkot", "Skt"],                      65),
    ("Hyderabad",   ["HYD", "hyderabad", "Hyd"],                    60),
    ("Bahawalpur",  ["BWP", "bahawalpur", "Bwp"],                   55),
    ("Sargodha",    ["SGD", "sargodha", "Sgd"],                     50),
    ("Sukkur",      ["SUK", "sukkur", "Suk"],                       40),
    ("Larkana",     ["LRK", "larkana", "Lrk"],                      35),
]

# Derived: flat list of canonical city names in config.CANONICAL_CITIES order
CANONICAL_CITY_NAMES: list[str] = [row[0] for row in CITY_DATA]

# Map every alias -> canonical name (used by cleaning; built at import time)
CITY_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canon, aliases, _ in CITY_DATA:
    CITY_ALIAS_TO_CANONICAL[canon.lower()] = canon
    for alias in aliases:
        CITY_ALIAS_TO_CANONICAL[alias.lower()] = canon

# ── Address component pools ───────────────────────────────────────────────────
STREET_TYPES: list[str] = [
    "Street", "St", "Road", "Rd", "Avenue", "Ave",
    "Lane", "Ln", "Block", "Sector", "Mohalla",
]

AREA_NAMES: list[str] = [
    "Gulberg", "DHA", "Model Town", "Johar Town", "Allama Iqbal Town",
    "Bahria Town", "Clifton", "North Nazimabad", "Gulshan", "Korangi",
    "Saddar", "Cantt", "Old City", "Wapda Town", "Askari",
    "F-7", "G-9", "I-8", "E-11", "H-13",
    "Satellite Town", "Chaklala Scheme", "Westridge",
]

# ── Gender raw codes ─────────────────────────────────────────────────────────
# Clean-gender -> dirty variants injected in raw data / duplicates
GENDER_VARIANTS: dict[str, list[str]] = {
    "M": ["Male", "male", "1", "m", "MALE", "Ml"],
    "F": ["Female", "female", "2", "f", "FEMALE", "Fm"],
}

# ── Blood groups ─────────────────────────────────────────────────────────────
BLOOD_GROUPS: list[str] = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ── Insurance tiers (ordinal) ─────────────────────────────────────────────────
# Order is important: None < Basic < Standard < Premium
INSURANCE_TIERS: list[str] = ["None", "Basic", "Standard", "Premium"]
# Frequency weights
INSURANCE_TIER_WEIGHTS: list[int] = [40, 30, 20, 10]
