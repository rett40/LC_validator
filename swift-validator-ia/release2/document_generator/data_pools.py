"""
data_pools.py — Données réalistes pour la génération de B/L synthétiques.
Ports, compagnies, marchandises, incoterms, etc.
"""

# ─────────────────────────────────────────────
#  Compagnies maritimes
# ─────────────────────────────────────────────

SHIPPING_LINES = {
    "msc": {
        "name": "MEDITERRANEAN SHIPPING COMPANY S.A.",
        "short": "MSC",
        "address": "12-14, Chemin Rieu, CH-1208 Geneva, Switzerland",
        "bl_prefix": "MSCUN",
        "color_primary": "#D42E12",
    },
    "maersk": {
        "name": "MAERSK LINE A/S",
        "short": "MAERSK",
        "address": "50 Esplanaden, DK-1098 Copenhagen K, Denmark",
        "bl_prefix": "MAEU",
        "color_primary": "#00243D",
    },
    "cmacgm": {
        "name": "CMA CGM S.A.",
        "short": "CMA CGM",
        "address": "4, Quai d'Arenc, 13002 Marseille, France",
        "bl_prefix": "CMAU",
        "color_primary": "#E30613",
    },
    "hapaglloyd": {
        "name": "HAPAG-LLOYD AKTIENGESELLSCHAFT",
        "short": "HAPAG-LLOYD",
        "address": "Ballindamm 25, D-20095 Hamburg, Germany",
        "bl_prefix": "HLCU",
        "color_primary": "#E05206",
    },
    "one": {
        "name": "OCEAN NETWORK EXPRESS PTE. LTD.",
        "short": "ONE",
        "address": "7 Straits View, #19-01 Marina One East Tower, Singapore 018936",
        "bl_prefix": "ONEY",
        "color_primary": "#E84393",
    },
    "evergreen": {
        "name": "EVERGREEN MARINE CORPORATION (TAIWAN) LTD.",
        "short": "EVERGREEN",
        "address": "166 Minsheng E. Road, Sec. 2, Taipei 10455, Taiwan",
        "bl_prefix": "EISU",
        "color_primary": "#009444",
    },
}

# ─────────────────────────────────────────────
#  Ports mondiaux (top 50 routes)
# ─────────────────────────────────────────────

PORTS = [
    # Asie
    {"name": "SHANGHAI", "country": "CHINA", "locode": "CNSHA", "region": "asia"},
    {"name": "NINGBO-ZHOUSHAN", "country": "CHINA", "locode": "CNNGB", "region": "asia"},
    {"name": "SHENZHEN", "country": "CHINA", "locode": "CNSZX", "region": "asia"},
    {"name": "GUANGZHOU", "country": "CHINA", "locode": "CNGUA", "region": "asia"},
    {"name": "TIANJIN", "country": "CHINA", "locode": "CNTXG", "region": "asia"},
    {"name": "QINGDAO", "country": "CHINA", "locode": "CNTAO", "region": "asia"},
    {"name": "HONG KONG", "country": "HONG KONG SAR", "locode": "HKHKG", "region": "asia"},
    {"name": "BUSAN", "country": "SOUTH KOREA", "locode": "KRPUS", "region": "asia"},
    {"name": "KAOHSIUNG", "country": "TAIWAN", "locode": "TWKHH", "region": "asia"},
    {"name": "SINGAPORE", "country": "SINGAPORE", "locode": "SGSIN", "region": "asia"},
    {"name": "TOKYO", "country": "JAPAN", "locode": "JPTYO", "region": "asia"},
    {"name": "YOKOHAMA", "country": "JAPAN", "locode": "JPYOK", "region": "asia"},
    {"name": "PORT KLANG", "country": "MALAYSIA", "locode": "MYPKG", "region": "asia"},
    {"name": "TANJUNG PELEPAS", "country": "MALAYSIA", "locode": "MYTPP", "region": "asia"},
    {"name": "LAEM CHABANG", "country": "THAILAND", "locode": "THLCH", "region": "asia"},
    {"name": "COLOMBO", "country": "SRI LANKA", "locode": "LKCMB", "region": "asia"},
    {"name": "JAKARTA", "country": "INDONESIA", "locode": "IDJKT", "region": "asia"},
    {"name": "MUMBAI", "country": "INDIA", "locode": "INBOM", "region": "asia"},
    {"name": "NHAVA SHEVA", "country": "INDIA", "locode": "INNSA", "region": "asia"},
    # Europe
    {"name": "ROTTERDAM", "country": "NETHERLANDS", "locode": "NLRTM", "region": "europe"},
    {"name": "ANTWERP", "country": "BELGIUM", "locode": "BEANR", "region": "europe"},
    {"name": "HAMBURG", "country": "GERMANY", "locode": "DEHAM", "region": "europe"},
    {"name": "FELIXSTOWE", "country": "UNITED KINGDOM", "locode": "GBFXT", "region": "europe"},
    {"name": "LE HAVRE", "country": "FRANCE", "locode": "FRLEH", "region": "europe"},
    {"name": "BREMERHAVEN", "country": "GERMANY", "locode": "DEBRV", "region": "europe"},
    {"name": "BARCELONA", "country": "SPAIN", "locode": "ESBCN", "region": "europe"},
    {"name": "VALENCIA", "country": "SPAIN", "locode": "ESVLC", "region": "europe"},
    {"name": "GENOA", "country": "ITALY", "locode": "ITGOA", "region": "europe"},
    {"name": "PIRAEUS", "country": "GREECE", "locode": "GRPIR", "region": "europe"},
    {"name": "GOTHENBURG", "country": "SWEDEN", "locode": "SEGOT", "region": "europe"},
    # Amériques
    {"name": "LOS ANGELES", "country": "USA", "locode": "USLAX", "region": "americas"},
    {"name": "LONG BEACH", "country": "USA", "locode": "USLGB", "region": "americas"},
    {"name": "NEW YORK", "country": "USA", "locode": "USNYC", "region": "americas"},
    {"name": "SAVANNAH", "country": "USA", "locode": "USSAV", "region": "americas"},
    {"name": "HOUSTON", "country": "USA", "locode": "USHOU", "region": "americas"},
    {"name": "SEATTLE", "country": "USA", "locode": "USSEA", "region": "americas"},
    {"name": "VANCOUVER", "country": "CANADA", "locode": "CAVAN", "region": "americas"},
    {"name": "SANTOS", "country": "BRAZIL", "locode": "BRSSZ", "region": "americas"},
    {"name": "CARTAGENA", "country": "COLOMBIA", "locode": "COCTG", "region": "americas"},
    {"name": "MANZANILLO", "country": "MEXICO", "locode": "MXZLO", "region": "americas"},
    # Moyen-Orient & Afrique
    {"name": "JEBEL ALI", "country": "UAE", "locode": "AEJEA", "region": "middleeast"},
    {"name": "SALALAH", "country": "OMAN", "locode": "OMSAL", "region": "middleeast"},
    {"name": "PORT SAID", "country": "EGYPT", "locode": "EGPSD", "region": "africa"},
    {"name": "DURBAN", "country": "SOUTH AFRICA", "locode": "ZADUR", "region": "africa"},
    {"name": "MOMBASA", "country": "KENYA", "locode": "KEMBA", "region": "africa"},
    {"name": "CASABLANCA", "country": "MOROCCO", "locode": "MACAS", "region": "africa"},
    {"name": "DAKAR", "country": "SENEGAL", "locode": "SNDKR", "region": "africa"},
    {"name": "LAGOS", "country": "NIGERIA", "locode": "NGAPP", "region": "africa"},
]

# ─────────────────────────────────────────────
#  Types de conteneurs
# ─────────────────────────────────────────────

CONTAINER_TYPES = [
    "20' DRY",
    "40' DRY",
    "40' HIGH CUBE",
    "20' REEFER",
    "40' REEFER",
    "20' OPEN TOP",
    "40' OPEN TOP",
    "20' FLAT RACK",
    "40' FLAT RACK",
    "45' HIGH CUBE",
    "20' TANK",
    "40' HIGH CUBE REEFER",
]

# ─────────────────────────────────────────────
#  Marchandises avec codes HS
# ─────────────────────────────────────────────

COMMODITIES = [
    {"name": "ELECTRONIC COMPONENTS AND PARTS", "hs_chapter": "85", "category": "electronics"},
    {"name": "TEXTILE FABRICS AND GARMENTS", "hs_chapter": "52", "category": "textile"},
    {"name": "AUTOMOTIVE PARTS AND ACCESSORIES", "hs_chapter": "87", "category": "automotive"},
    {"name": "INDUSTRIAL MACHINERY AND EQUIPMENT", "hs_chapter": "84", "category": "machinery"},
    {"name": "PLASTIC GOODS AND ARTICLES", "hs_chapter": "39", "category": "plastics"},
    {"name": "FURNITURE AND PARTS THEREOF", "hs_chapter": "94", "category": "furniture"},
    {"name": "FOOTWEAR OF ALL TYPES", "hs_chapter": "64", "category": "footwear"},
    {"name": "TOYS, GAMES AND SPORTS EQUIPMENT", "hs_chapter": "95", "category": "toys"},
    {"name": "HOUSEHOLD APPLIANCES", "hs_chapter": "85", "category": "appliances"},
    {"name": "STEEL PRODUCTS AND STRUCTURES", "hs_chapter": "73", "category": "steel"},
    {"name": "CHEMICAL PRODUCTS NEC", "hs_chapter": "38", "category": "chemicals"},
    {"name": "AGRICULTURAL PRODUCTS AND FOODSTUFFS", "hs_chapter": "07", "category": "food"},
    {"name": "PAPER AND PAPERBOARD ARTICLES", "hs_chapter": "48", "category": "paper"},
    {"name": "RUBBER PRODUCTS AND TYRES", "hs_chapter": "40", "category": "rubber"},
    {"name": "OPTICAL AND PHOTOGRAPHIC INSTRUMENTS", "hs_chapter": "90", "category": "optical"},
    {"name": "MEDICAL DEVICES AND EQUIPMENT", "hs_chapter": "90", "category": "medical"},
    {"name": "BUILDING MATERIALS", "hs_chapter": "68", "category": "construction"},
    {"name": "PERSONAL CARE PRODUCTS", "hs_chapter": "33", "category": "cosmetics"},
    {"name": "CERAMIC TILES AND SANITARY WARE", "hs_chapter": "69", "category": "ceramics"},
    {"name": "COPPER AND COPPER PRODUCTS", "hs_chapter": "74", "category": "metals"},
    {"name": "WOOD PRODUCTS AND TIMBER", "hs_chapter": "44", "category": "wood"},
    {"name": "SEAFOOD AND FROZEN FISH PRODUCTS", "hs_chapter": "03", "category": "food"},
    {"name": "BEVERAGES AND SPIRITS", "hs_chapter": "22", "category": "food"},
    {"name": "SOLAR PANELS AND PHOTOVOLTAIC CELLS", "hs_chapter": "85", "category": "energy"},
    {"name": "LITHIUM-ION BATTERIES", "hs_chapter": "85", "category": "energy"},
]

# ─────────────────────────────────────────────
#  Incoterms 2020
# ─────────────────────────────────────────────

INCOTERMS = [
    "EXW",   # Ex Works
    "FCA",   # Free Carrier
    "FAS",   # Free Alongside Ship
    "FOB",   # Free On Board
    "CFR",   # Cost and Freight
    "CIF",   # Cost, Insurance and Freight
    "CPT",   # Carriage Paid To
    "CIP",   # Carriage and Insurance Paid To
    "DAP",   # Delivered At Place
    "DPU",   # Delivered at Place Unloaded
    "DDP",   # Delivered Duty Paid
]

# ─────────────────────────────────────────────
#  Freight Terms
# ─────────────────────────────────────────────

FREIGHT_TERMS = [
    "PREPAID",
    "COLLECT",
    "AS ARRANGED",
    "FREIGHT PREPAID",
    "FREIGHT COLLECT",
    "FREIGHT PREPAID AT ORIGIN",
    "FREIGHT COLLECT AT DESTINATION",
]

# ─────────────────────────────────────────────
#  Notify Parties types
# ─────────────────────────────────────────────

NOTIFY_PARTIES = [
    "SAME AS CONSIGNEE",
    "NOTIFY FREIGHT FORWARDER",
    "NOTIFY CUSTOMS BROKER",
    "NOTIFY BANK",
]
