"""
Bill of Lading Synthetic Dataset Generator — VERSION CORRIGÉE
=============================================================
Corrections apportées :
  1. ETA toujours >= ETD (cohérence des dates)
  2. Adresses ASCII uniquement (plus de caractères ■■■)
  3. Séparateur correct dans la ligne cargo (TYPE | PKGS bien séparés)
  4. Bordures de colonnes dans le tableau cargo
  5. Faker limité à en_US pour éviter les encodages non-ASCII
  6. place_of_receipt / place_of_delivery cohérents avec POL/POD
  7. Numéro BL annoté correctement dans la bande d'en-tête

Usage:
    python generate_bl.py --count 400 --output ./dataset --formats all
    python generate_bl.py --count 100 --output ./dataset --formats msc,maersk
"""

import json
import random
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# FIX #5 : Faker uniquement en_US → caractères ASCII garantis
from faker import Faker
from data_pools import (
    PORTS, CONTAINER_TYPES, COMMODITIES, SHIPPING_LINES,
    INCOTERMS, NOTIFY_PARTIES, FREIGHT_TERMS
)

fake = Faker('en_US')   # ← était ['en_US', 'fr_FR', 'de_DE', 'zh_CN']
# Module-level seed removed — use generate_dataset(seed=...) for reproducibility


# ─────────────────────────────────────────────
#  Utilitaires de données aléatoires
# ─────────────────────────────────────────────

def random_bl_number(prefix: str) -> str:
    letters = ''.join(random.choices('ABCDEFGHJKLMNPRSTUVWXYZ', k=4))
    digits = ''.join(random.choices('0123456789', k=9))
    return f"{prefix}{letters}{digits}"


def random_container_number() -> str:
    owners = ['MSCU', 'MAEU', 'CMAU', 'HLCU', 'OOLU', 'EISU', 'TRHU', 'TCKU']
    owner = random.choice(owners)
    digits = ''.join(random.choices('0123456789', k=6))
    check = random.randint(0, 9)
    return f"{owner}{digits}{check}"


def random_vessel() -> dict:
    vessels = [
        {"name": "MSC OSCAR",             "voyage": f"FE{random.randint(100,999)}N"},
        {"name": "MAERSK HARTFORD",        "voyage": f"AE{random.randint(10,99)}{random.choice('NESW')}"},
        {"name": "CMA CGM JULES VERNE",    "voyage": f"0{random.randint(100,999)}{random.choice('NESW')}"},
        {"name": "HAPAG LLOYD EXPRESS",    "voyage": f"AL{random.randint(1,9)}{random.randint(100,999)}"},
        {"name": "EVER GIVEN",             "voyage": f"E-{random.randint(100,999)}"},
        {"name": "MSC GULSUN",             "voyage": f"FE{random.randint(100,999)}S"},
        {"name": "OOCL HONG KONG",         "voyage": f"0{random.randint(100,999)}{random.choice('NESW')}"},
        {"name": "COSCO SHIPPING UNIVERSE","voyage": f"CS{random.randint(100,999)}{random.choice('NESW')}"},
    ]
    return random.choice(vessels)


# FIX #1 : ETD et ETA cohérents (ETA >= ETD + délai transit)
def random_date_pair() -> tuple:
    """Retourne (ETD, ETA) avec ETA toujours > ETD."""
    base = datetime.now()
    etd_delta = timedelta(days=random.randint(5, 45))
    transit_days = random.randint(14, 35)   # 2–5 semaines de transit
    etd = base + etd_delta
    eta = etd + timedelta(days=transit_days)
    fmt = "%d %b %Y"
    return etd.strftime(fmt).upper(), eta.strftime(fmt).upper()


def random_issue_date() -> str:
    base = datetime.now()
    delta = timedelta(days=random.randint(-10, -1))
    return (base + delta).strftime("%d %b %Y").upper()


def random_weight() -> dict:
    gross_kg = random.uniform(5000, 28000)
    tare_kg = random.uniform(2000, 4000)
    net_kg = gross_kg - tare_kg
    return {
        "gross_kg": f"{gross_kg:,.2f}",
        "net_kg": f"{net_kg:,.2f}",
        "gross_lbs": f"{gross_kg * 2.20462:,.2f}",
        "net_lbs": f"{net_kg * 2.20462:,.2f}",
    }


def random_volume() -> str:
    return f"{random.uniform(20, 68):.3f} CBM"


def random_packages() -> dict:
    types = ["PALLETS", "CARTONS", "PACKAGES", "BAGS", "DRUMS", "BOXES", "CRATES"]
    count = random.randint(10, 500)
    pkg_type = random.choice(types)
    return {"count": str(count), "type": pkg_type, "description": f"{count} {pkg_type}"}


def random_container_entry() -> dict:
    """Returns a single container dict with all cargo-related fields."""
    commodity = random.choice(COMMODITIES)
    weight = random_weight()
    packages = random_packages()
    return {
        "container_number": random_container_number(),
        "seal_number":       ''.join(random.choices('ABCDEFGH0123456789', k=8)),
        "container_type":    random.choice(CONTAINER_TYPES),
        "commodity":         commodity["name"],
        "packages":          packages,
        "gross_weight_kg":   weight["gross_kg"],
        "net_weight_kg":     weight["net_kg"],
        "measurement":       random_volume(),
    }


def random_containers() -> list:
    """Returns 1–3 containers with weighted probability [50%, 35%, 15%]."""
    n = random.choices([1, 2, 3], weights=[50, 35, 15], k=1)[0]
    return [random_container_entry() for _ in range(n)]


# FIX #2 : Adresses ASCII uniquement, format structuré propre
def random_company_address() -> dict:
    """Génère une adresse 100% ASCII compatible PDF."""
    suffixes = ["LTD", "LLC", "INC", "GMBH", "S.A.", "CO.", "CORP", "PLC", "AG", "BV"]
    adjectives = ["GLOBAL", "INTERNATIONAL", "PACIFIC", "ATLANTIC", "EURO", "TRANS",
                  "FIRST", "PREMIER", "CENTRAL", "UNITED", "APEX", "DELTA"]
    nouns = ["TRADING", "LOGISTICS", "SHIPPING", "COMMERCE", "INDUSTRIES",
             "SOLUTIONS", "ENTERPRISES", "GROUP", "PARTNERS", "SUPPLY CHAIN"]
    company = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(suffixes)}"

    streets = ["MAIN STREET", "HARBOR ROAD", "INDUSTRIAL AVENUE", "TRADE PARK",
               "COMMERCE BOULEVARD", "PORT ACCESS ROAD", "MARKET STREET"]
    cities_countries = [
        ("NEW YORK", "USA", "NY 10001"),
        ("LONDON", "UNITED KINGDOM", "EC1A 1BB"),
        ("HAMBURG", "GERMANY", "20095"),
        ("ROTTERDAM", "NETHERLANDS", "3011 AA"),
        ("SINGAPORE", "SINGAPORE", "018989"),
        ("HONG KONG", "HONG KONG SAR", "999077"),
        ("DUBAI", "UAE", "00000"),
        ("SHANGHAI", "CHINA", "200120"),
        ("LOS ANGELES", "USA", "CA 90001"),
        ("ANTWERP", "BELGIUM", "2000"),
        ("BARCELONA", "SPAIN", "08003"),
        ("BUSAN", "SOUTH KOREA", "48940"),
    ]
    city, country, zipcode = random.choice(cities_countries)
    street_num = random.randint(1, 999)
    street = random.choice(streets)

    return {
        "name": company,
        "address": f"{street_num} {street}",
        "city": city,
        "country": country,
        "zip": zipcode,
        "full": f"{company}\n{street_num} {street}\n{city}, {country} {zipcode}"
    }


def random_port_pair() -> tuple:
    """Retourne (POL, POD) avec ville de réception/livraison cohérente."""
    routes = [
        ("SHANGHAI, CHINA",       "ROTTERDAM, NETHERLANDS"),
        ("NINGBO, CHINA",         "HAMBURG, GERMANY"),
        ("SHENZHEN, CHINA",       "LOS ANGELES, USA"),
        ("GUANGZHOU, CHINA",      "ANTWERP, BELGIUM"),
        ("BUSAN, SOUTH KOREA",    "FELIXSTOWE, UK"),
        ("SINGAPORE",             "LE HAVRE, FRANCE"),
        ("HONG KONG",             "NEW YORK, USA"),
        ("TIANJIN, CHINA",        "VALENCIA, SPAIN"),
        ("KAOHSIUNG, TAIWAN",     "GENOA, ITALY"),
        ("TOKYO, JAPAN",          "LONG BEACH, USA"),
        ("MUMBAI, INDIA",         "JEBEL ALI, UAE"),
        ("COLOMBO, SRI LANKA",    "PIRAEUS, GREECE"),
    ]
    return random.choice(routes)


def build_bl_data(carrier_key: str) -> dict:
    """Construit le dictionnaire complet de données pour un B/L (multi-container)."""
    carrier = SHIPPING_LINES[carrier_key]
    vessel = random_vessel()
    port_load, port_discharge = random_port_pair()
    shipper = random_company_address()
    consignee = random_company_address()
    notify = random_company_address()

    # Multi-container support (1-3 containers per BL)
    containers = random_containers()
    first = containers[0]

    # Aggregate totals across all containers
    total_gross_kg = sum(float(c["gross_weight_kg"].replace(",", "")) for c in containers)
    total_net_kg   = sum(float(c["net_weight_kg"].replace(",", ""))   for c in containers)
    total_cbm      = sum(float(c["measurement"].split()[0])            for c in containers)

    # FIX #1 : ETD/ETA cohérents
    etd, eta = random_date_pair()
    issue_date = random_issue_date()

    # FIX #6 : place_of_receipt / place_of_delivery extraits proprement
    pol_city = port_load.split(",")[0].strip()
    pod_city = port_discharge.split(",")[0].strip()
    service_options = ["CY", "CFS"]
    place_of_receipt = f"{pol_city} {random.choice(service_options)}"
    place_of_delivery = f"{pod_city} {random.choice(service_options)}"

    return {
        "carrier": carrier,
        "carrier_key": carrier_key,
        "bl_number": random_bl_number(carrier["bl_prefix"]),

        "shipper": shipper,
        "consignee": consignee,
        "notify_party": notify,

        "vessel": vessel["name"],
        "voyage": vessel["voyage"],
        "port_of_loading": port_load,
        "port_of_discharge": port_discharge,
        "place_of_receipt": place_of_receipt,
        "place_of_delivery": place_of_delivery,
        "etd": etd,
        "eta": eta,
        "issue_date": issue_date,
        "issue_place": random.choice(["SHANGHAI", "SINGAPORE", "ROTTERDAM", "HAMBURG"]),

        # Top-level single-container fields — backward compat for invoice/packing_list generators
        "container_number": first["container_number"],
        "container_type":   first["container_type"],
        "seal_number":      first["seal_number"],
        "commodity":        first["commodity"],
        "packages":         first["packages"],
        "gross_weight_kg":  f"{total_gross_kg:,.2f}",
        "net_weight_kg":    f"{total_net_kg:,.2f}",
        "gross_weight_lbs": f"{total_gross_kg * 2.20462:,.2f}",
        "measurement":      f"{total_cbm:.3f} CBM",

        # Multi-container data used by renderers
        "containers":   containers,
        "n_containers": len(containers),

        "freight_terms": random.choice(FREIGHT_TERMS),
        "incoterm": random.choice(INCOTERMS),
        "number_of_originals": random.choice(["THREE (3)", "TWO (2)", "ONE (1)"]),
        "service_type": random.choice(["CY/CY", "CY/CFS", "CFS/CY", "CFS/CFS"]),
        "marks_numbers": f"{first['container_number']}\n{first['seal_number']}\n{pod_city}",
    }


# ─────────────────────────────────────────────
#  Classe de base
# ─────────────────────────────────────────────

# Pool de fontes standard ReportLab — simule la diversité des vrais docs
_FONT_PAIRS = [
    ("Helvetica",   "Helvetica-Bold"),
    ("Helvetica",   "Helvetica-Bold"),   # sur-représenté : le plus courant
    ("Times-Roman", "Times-Bold"),
    ("Times-Roman", "Times-Bold"),
    ("Courier",     "Courier-Bold"),
]


class BLRenderer:
    PAGE_W, PAGE_H = A4
    MARGIN = 1.0 * cm

    def __init__(self, output_path: str, data: dict):
        self.path = output_path
        self.data = data
        self.c = canvas.Canvas(output_path, pagesize=A4)
        self.annotations = []
        # Choisir une paire de fontes aléatoirement à chaque document
        self.FONT, self.FONT_BOLD = random.choice(_FONT_PAIRS)

    def _annotate(self, field_name, x, y, w, h, value):
        self.annotations.append({
            "field": field_name,
            "bbox": [round(x, 2), round(self.PAGE_H - y - h, 2), round(w, 2), round(h, 2)],
            "value": value,
            "page": 1
        })

    def _draw_text(self, x, y, text, font=None, size=8, color=colors.black):
        font = font or self.FONT
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        self.c.drawString(x, y, str(text))

    def _draw_text_field(self, field_name, x, y, w, h, value,
                         font=None, size=8, align="left", color=colors.black):
        font = font or self.FONT
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        text = str(value).replace('\n', ' ')
        max_w = max(w - 4, 1)
        while self.c.stringWidth(text, font, size) > max_w and len(text) > 4:
            text = text[:-1]
        ty = y - h / 2 - size / 3
        if align == "center":
            self.c.drawCentredString(x + w / 2, ty, text)
        elif align == "right":
            self.c.drawRightString(x + w - 2, ty, text)
        else:
            self.c.drawString(x + 2, ty, text)
        self._annotate(field_name, x, y - h, w, h, value)

    def _draw_multiline_field(self, field_name, x, y, w, h,
                               value, font=None, size=7.5):
        font = font or self.FONT
        self.c.setFont(font, size)
        self.c.setFillColor(colors.black)
        lines = str(value).split('\n')
        line_h = size * 1.4
        current_y = y - line_h
        bottom_y = y - h
        for line in lines:
            if current_y < bottom_y + size:
                break
            while self.c.stringWidth(line, font, size) > w - 4 and len(line) > 4:
                line = line[:-1]
            self.c.drawString(x + 2, current_y, line)
            current_y -= line_h
        self._annotate(field_name, x, y - h, w, h, value)

    # FIX #4 : Dessin des lignes de séparation de colonnes dans le tableau cargo
    def _draw_cargo_table(self, x, y, total_w, col_widths, headers, values,
                          header_h=0.5*cm, row_h=1.8*cm,
                          header_bg=colors.HexColor("#003087"),
                          border_color=colors.HexColor("#CCCCCC")):
        c = self.c
        # En-tête
        c.setFillColor(header_bg)
        c.rect(x, y - header_h, total_w, header_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(self.FONT_BOLD, 6.5)
        cx = x
        for hdr, cw in zip(headers, col_widths):
            c.drawString(cx + 2, y - header_h + 3, hdr)
            cx += cw

        # Ligne de données
        c.setFillColor(colors.white)
        c.setStrokeColor(border_color)
        c.rect(x, y - header_h - row_h, total_w, row_h, fill=1, stroke=1)

        # Lignes verticales de séparation
        cx = x
        for cw in col_widths[:-1]:
            cx += cw
            c.setStrokeColor(border_color)
            c.line(cx, y - header_h, cx, y - header_h - row_h)

        # Valeurs
        c.setFont(self.FONT, 7)
        c.setFillColor(colors.black)
        cx = x
        annotations_to_make = []
        for val, cw in zip(values, col_widths):
            text = str(val)
            ty = y - header_h - 0.5*cm
            while c.stringWidth(text, self.FONT, 7) > cw - 4 and len(text) > 4:
                text = text[:-1]
            c.drawString(cx + 2, ty, text)
            annotations_to_make.append((cx, cw))
            cx += cw

        return (x, y - header_h - row_h, total_w, row_h)

    def _draw_multi_cargo_table(self, x, y, total_w, col_widths, headers, rows,
                                 header_h=0.5*cm, row_h=1.4*cm,
                                 header_bg=colors.HexColor("#003087"),
                                 border_color=colors.HexColor("#CCCCCC")):
        """
        Multi-row cargo table for 1-3 containers.
        rows: list of lists of strings (one inner list per container).
        Returns (bottom_y, row_top_ys, row_h) for per-container annotations.
        """
        c = self.c
        c.setFillColor(header_bg)
        c.rect(x, y - header_h, total_w, header_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(self.FONT_BOLD, 6.5)
        cx = x
        for hdr, cw in zip(headers, col_widths):
            c.drawString(cx + 2, y - header_h + 3, hdr)
            cx += cw

        row_top_ys = []
        current_top = y - header_h
        even_bg = colors.white
        odd_bg  = colors.HexColor("#F5F5F5")

        for idx, row_vals in enumerate(rows):
            c.setFillColor(even_bg if idx % 2 == 0 else odd_bg)
            c.setStrokeColor(border_color)
            c.rect(x, current_top - row_h, total_w, row_h, fill=1, stroke=1)
            cx_sep = x
            for cw in col_widths[:-1]:
                cx_sep += cw
                c.line(cx_sep, current_top, cx_sep, current_top - row_h)
            c.setFont(self.FONT, 7)
            c.setFillColor(colors.black)
            cx = x
            for val, cw in zip(row_vals, col_widths):
                text = str(val)
                while c.stringWidth(text, self.FONT, 7) > cw - 4 and len(text) > 4:
                    text = text[:-1]
                c.drawString(cx + 2, current_top - row_h / 2 - 3, text)
                cx += cw
            row_top_ys.append(current_top)
            current_top -= row_h

        return current_top, row_top_ys, row_h

    def render(self):
        raise NotImplementedError

    def save(self):
        self.c.save()
        return self.annotations


# ─────────────────────────────────────────────
#  MSC Renderer
# ─────────────────────────────────────────────

class MSCRenderer(BLRenderer):
    PRIMARY   = colors.HexColor("#D42E12")
    SECONDARY = colors.HexColor("#1A3C5E")
    LIGHT_BG  = colors.HexColor("#F5F5F5")
    BORDER    = colors.HexColor("#CCCCCC")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # En-tête rouge
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.8*cm, W, 2.8*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(M, H - 1.8*cm, "MSC")
        c.setFont("Helvetica", 9)
        c.drawString(M, H - 2.3*cm, "MEDITERRANEAN SHIPPING COMPANY S.A.")
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(W - M, H - 1.5*cm, "BILL OF LADING")
        c.setFont("Helvetica", 8)
        c.drawRightString(W - M, H - 2.1*cm, "NON-NEGOTIABLE UNLESS CONSIGNED TO ORDER")

        # Bande bleue BL info
        c.setFillColor(self.SECONDARY)
        c.rect(0, H - 3.4*cm, W, 0.6*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 3.15*cm, f"B/L NUMBER:  {d['bl_number']}")
        c.drawRightString(W - M, H - 3.15*cm,
            f"DATE OF ISSUE: {d['issue_date']}  |  PLACE: {d['issue_place']}")
        # FIX #7 : annotation du BL number dans la bande en-tête
        self._annotate("bl_number", M, H - 3.4*cm, 8*cm, 0.6*cm, d['bl_number'])

        top = H - 3.7*cm
        row_h = 3.0*cm
        half_w = (W - 2*M) / 2
        label_h = 12

        def box(x, y, w, h, label, light=False):
            c.setFillColor(self.LIGHT_BG if light else colors.white)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h, fill=1, stroke=1)
            c.setFillColor(self.SECONDARY)
            c.setFont("Helvetica-Bold", 6.5)
            c.drawString(x + 2, y - 9, label)

        # Shipper / Consignee
        y = top
        box(M, y, half_w, row_h, "SHIPPER / EXPORTER", light=True)
        self._draw_multiline_field("shipper_name", M, y - label_h, half_w, row_h - label_h,
            d['shipper']['full'])

        box(M + half_w, y, half_w, row_h, "CONSIGNEE")
        self._draw_multiline_field("consignee_name", M + half_w, y - label_h, half_w, row_h - label_h,
            d['consignee']['full'])

        # Notify / Routing
        y = top - row_h
        box(M, y, half_w, row_h, "NOTIFY PARTY", light=True)
        self._draw_multiline_field("notify_party", M, y - label_h, half_w, row_h - label_h,
            d['notify_party']['full'])

        quarter = half_w / 2
        box(M + half_w, y, quarter, row_h / 2, "VESSEL")
        self._draw_text_field("vessel", M + half_w, y - label_h, quarter, row_h / 2 - label_h,
            d['vessel'], size=7)
        box(M + half_w + quarter, y, quarter, row_h / 2, "VOYAGE NO.")
        self._draw_text_field("voyage", M + half_w + quarter, y - label_h, quarter, row_h / 2 - label_h,
            d['voyage'], size=7)

        y2 = y - row_h / 2
        box(M + half_w, y2, quarter, row_h / 2, "PORT OF LOADING")
        self._draw_text_field("port_of_loading", M + half_w, y2 - label_h, quarter, row_h / 2 - label_h,
            d['port_of_loading'], size=6.5)
        box(M + half_w + quarter, y2, quarter, row_h / 2, "PORT OF DISCHARGE")
        self._draw_text_field("port_of_discharge", M + half_w + quarter, y2 - label_h, quarter, row_h / 2 - label_h,
            d['port_of_discharge'], size=6.5)

        # Routing 2e rangée
        y = top - 2 * row_h
        q4 = (W - 2*M) / 4
        for i, (label, val) in enumerate([
            ("PLACE OF RECEIPT", d['place_of_receipt']),
            ("PLACE OF DELIVERY", d['place_of_delivery']),
            ("ETD", d['etd']),
            ("ETA", d['eta']),
        ]):
            rx = M + i * q4
            box(rx, y, q4, 1.2*cm, label)
            self._draw_text_field(label.lower().replace(" ", "_"), rx, y - label_h, q4, 1.2*cm - label_h,
                val, size=6.5)

        # Tableau cargo multi-conteneurs
        y -= 1.2*cm + 0.3*cm
        headers = ["CONTAINER NO.", "SEAL NO.", "TYPE", "PACKAGES", "DESCRIPTION", "GROSS WEIGHT", "CBM"]
        cws = [3.0*cm, 2.0*cm, 2.3*cm, 2.0*cm, 4.7*cm, 2.8*cm, 2.2*cm]
        total_w = W - 2*M
        rows = [
            [c['container_number'], c['seal_number'], c['container_type'],
             c['packages']['description'], c['commodity'],
             f"{c['gross_weight_kg']} KG", c['measurement']]
            for c in d['containers']
        ]
        bottom_y, row_top_ys, rh = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows, header_bg=self.SECONDARY)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,             rty - rh, cws[0], rh, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],    rty - rh, cws[1], rh, cont['seal_number'])
            self._annotate("container_type",   M + sum(cws[:2]), rty - rh, cws[2], rh, cont['container_type'])
            self._annotate("gross_weight_kg",  M + sum(cws[:5]), rty - rh, cws[5], rh, cont['gross_weight_kg'])
            self._annotate("measurement",      M + sum(cws[:6]), rty - rh, cws[6], rh, cont['measurement'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (label, val) in enumerate([
            ("FREIGHT & CHARGES", d['freight_terms']),
            ("NUMBER OF ORIGINAL B/L", d['number_of_originals']),
            ("SERVICE TYPE", d['service_type']),
        ]):
            fx = M + i * thirds
            box(fx, y, thirds, 1.5*cm, label)
            self._draw_text_field(label.lower().replace(" ", "_"),
                fx, y - label_h, thirds, 1.5*cm - label_h, val, size=8)

        # Footer
        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(W / 2, 0.35*cm,
            "MSC Mediterranean Shipping Company S.A. | 12-14, Ch. Rieu, CH-1208 Geneva, Switzerland | www.msc.com")


# ─────────────────────────────────────────────
#  Maersk Renderer
# ─────────────────────────────────────────────

class MaerskRenderer(BLRenderer):
    PRIMARY  = colors.HexColor("#00243D")
    ACCENT   = colors.HexColor("#009BDE")
    LIGHT_BG = colors.HexColor("#EEF4F8")
    BORDER   = colors.HexColor("#B0C4D4")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # En-tête
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 3.0*cm, W, 3.0*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 3.0*cm, 0.8*cm, 3.0*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(1.2*cm, H - 1.7*cm, "MAERSK")
        c.setFont("Helvetica", 8)
        c.drawString(1.2*cm, H - 2.2*cm, "MAERSK LINE A/S")
        c.drawString(1.2*cm, H - 2.6*cm, "50 ESPLANADEN · DK-1098 COPENHAGEN K · DENMARK")
        c.setFont("Helvetica-Bold", 16)
        c.drawRightString(W - M, H - 1.6*cm, "BILL OF LADING")
        c.setFont("Helvetica", 8)
        c.drawRightString(W - M, H - 2.1*cm, "COMBINED TRANSPORT")
        c.drawRightString(W - M, H - 2.6*cm, f"B/L No: {d['bl_number']}")
        # FIX #7
        self._annotate("bl_number", W / 2, H - 3.0*cm, W / 2 - M, 0.55*cm, d['bl_number'])

        top = H - 3.3*cm
        col_w = (W - 2*M) / 2
        label_h = 0.42 * cm

        def fbox(x, y, w, h, label, val, multiline=False, fs=7.5):
            field_name = label.lower().replace(" ", "_")
            content_top = y - label_h
            content_h = h - label_h
            c.setFillColor(self.LIGHT_BG)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h, fill=1, stroke=1)
            c.setFillColor(self.PRIMARY)
            c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - 0.28*cm, label)
            c.setFillColor(colors.black)
            if multiline:
                self._draw_multiline_field(field_name,
                    x, content_top, w, content_h, val, size=fs)
            else:
                self._draw_text_field(field_name,
                    x, content_top, w, content_h, val, size=fs)

        row_h = 2.8*cm
        fbox(M, top, col_w, row_h, "SHIPPER", d['shipper']['full'], multiline=True)
        fbox(M + col_w, top, col_w, row_h, "CONSIGNEE", d['consignee']['full'], multiline=True)

        y = top - row_h
        fbox(M, y, col_w, row_h, "NOTIFY PARTY", d['notify_party']['full'], multiline=True)

        quarter = col_w / 2
        fbox(M + col_w, y, quarter, row_h / 2, "VESSEL", d['vessel'])
        fbox(M + col_w + quarter, y, quarter, row_h / 2, "VOYAGE NO.", d['voyage'])
        fbox(M + col_w, y - row_h / 2, quarter, row_h / 2, "PORT OF LOADING", d['port_of_loading'], fs=6.5)
        fbox(M + col_w + quarter, y - row_h / 2, quarter, row_h / 2, "PORT OF DISCHARGE", d['port_of_discharge'], fs=6.5)

        y -= row_h
        q4 = (W - 2*M) / 4
        fbox(M, y, q4, 1.2*cm, "PLACE OF RECEIPT", d['place_of_receipt'])
        fbox(M + q4, y, q4, 1.2*cm, "PLACE OF DELIVERY", d['place_of_delivery'])
        fbox(M + 2*q4, y, q4, 1.2*cm, "ETD", d['etd'])
        fbox(M + 3*q4, y, q4, 1.2*cm, "ETA", d['eta'])

        y -= 1.2*cm + 0.2*cm
        headers = ["CNTR NO. / SEAL", "TYPE", "QTY & PACKAGE", "DESCRIPTION OF GOODS",
                   "GROSS WEIGHT (KG)", "MEASUREMENT"]
        cws = [3.5*cm, 1.8*cm, 2.8*cm, 5.0*cm, 3.2*cm, 2.7*cm]
        total_w = W - 2*M
        rows = [
            [f"{c['container_number']} / {c['seal_number']}",
             c['container_type'], c['packages']['description'],
             c['commodity'], f"{c['gross_weight_kg']} KG", c['measurement']]
            for c in d['containers']
        ]
        bottom_y, row_top_ys, rh = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows, header_bg=self.PRIMARY)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,             rty - rh, cws[0], rh, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],    rty - rh, cws[1], rh, cont['seal_number'])
            self._annotate("gross_weight_kg",  M + sum(cws[:4]), rty - rh, cws[4], rh, cont['gross_weight_kg'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (label, val) in enumerate([
            ("FREIGHT PAYABLE AT", d['freight_terms']),
            ("NO. OF ORIGINAL B/L", d['number_of_originals']),
            ("DATE AND PLACE OF ISSUE", f"{d['issue_date']}, {d['issue_place']}"),
        ]):
            fbox(M + i * thirds, y, thirds, 1.5*cm, label, val)

        c.setFillColor(self.ACCENT)
        c.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W / 2, 0.28*cm,
            "Maersk Line A/S | Registered in Denmark | CVR 56314728 | www.maersk.com")


# ─────────────────────────────────────────────
#  CMA CGM Renderer
# ─────────────────────────────────────────────

class CMACGMRenderer(BLRenderer):
    PRIMARY   = colors.HexColor("#E30613")
    SECONDARY = colors.HexColor("#003087")
    STRIPE    = colors.HexColor("#F7F7F7")
    BORDER    = colors.HexColor("#CCCCCC")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # En-tête
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 1.0*cm, W, 1.0*cm, fill=1, stroke=0)
        c.setFillColor(self.SECONDARY)
        c.rect(0, H - 2.5*cm, W, 1.5*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M, H - 2.0*cm, "CMA CGM")
        c.setFont("Helvetica", 8)
        c.drawString(M, H - 2.35*cm, "THE FRENCH LEADER IN CONTAINER SHIPPING")
        c.setFont("Helvetica-Bold", 12)
        c.drawRightString(W - M, H - 1.85*cm, "BILL OF LADING")
        c.setFont("Helvetica", 7.5)
        c.drawRightString(W - M, H - 2.25*cm, "ORIGINAL — NOT NEGOTIABLE")

        # Bande ID
        c.setFillColor(self.STRIPE)
        c.rect(0, H - 3.2*cm, W, 0.7*cm, fill=1, stroke=0)
        c.setFillColor(self.SECONDARY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 2.85*cm, f"BL N°: {d['bl_number']}")
        c.drawString(W / 2, H - 2.85*cm, f"DATE: {d['issue_date']}")
        c.drawRightString(W - M, H - 2.85*cm, f"PLACE: {d['issue_place']}")
        self._annotate("bl_number", M, H - 3.2*cm, 8*cm, 0.7*cm, d['bl_number'])

        top = H - 3.5*cm
        c.setStrokeColor(self.BORDER)
        label_h = 0.38 * cm

        def lbox(x, y, w, h, label, val, multiline=False, fs=7.5):
            field_name = label.lower().replace(" ", "_")
            content_top = y - label_h
            content_h = h - label_h
            c.setFillColor(self.SECONDARY)
            c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - 0.25*cm, label)
            c.setFillColor(self.STRIPE)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, content_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            if multiline:
                self._draw_multiline_field(field_name,
                    x, content_top, w, content_h, val, size=fs)
            else:
                self._draw_text_field(field_name,
                    x, content_top, w, content_h, val, size=fs)

        half = (W - 2*M) / 2
        rh = 2.8*cm

        lbox(M, top, half, rh, "SHIPPER", d['shipper']['full'], multiline=True)
        lbox(M + half, top, half, rh, "CONSIGNEE (TO ORDER OF)", d['consignee']['full'], multiline=True)

        y = top - rh
        lbox(M, y, half, rh, "NOTIFY PARTY", d['notify_party']['full'], multiline=True)

        quarter = half / 2
        lbox(M + half, y, quarter, rh / 2, "VESSEL", d['vessel'])
        lbox(M + half + quarter, y, quarter, rh / 2, "VOYAGE", d['voyage'])
        lbox(M + half, y - rh / 2, quarter, rh / 2, "PORT OF LOADING", d['port_of_loading'], fs=6.5)
        lbox(M + half + quarter, y - rh / 2, quarter, rh / 2, "PORT OF DISCHARGE", d['port_of_discharge'], fs=6.5)

        y -= rh
        q4 = (W - 2*M) / 4
        lbox(M, y, q4, 1.2*cm, "PLACE OF RECEIPT", d['place_of_receipt'])
        lbox(M + q4, y, q4, 1.2*cm, "PLACE OF DELIVERY", d['place_of_delivery'])
        lbox(M + 2*q4, y, q4, 1.2*cm, "ETD", d['etd'])
        lbox(M + 3*q4, y, q4, 1.2*cm, "ETA", d['eta'])

        y -= 1.2*cm + 0.3*cm
        headers = ["CONTAINER NO.", "SEAL NO.", "TYPE", "PKGS", "DESCRIPTION", "GROSS WT (KG)", "CBM"]
        cws = [3.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 4.8*cm, 2.8*cm, 2.6*cm]
        total_w = W - 2*M
        rows = [
            [c['container_number'], c['seal_number'], c['container_type'],
             c['packages']['description'], c['commodity'],
             f"{c['gross_weight_kg']} KG", c['measurement']]
            for c in d['containers']
        ]
        bottom_y, row_top_ys, rh = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows, header_bg=self.PRIMARY)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,             rty - rh, cws[0], rh, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],    rty - rh, cws[1], rh, cont['seal_number'])
            self._annotate("container_type",   M + sum(cws[:2]), rty - rh, cws[2], rh, cont['container_type'])
            self._annotate("gross_weight_kg",  M + sum(cws[:5]), rty - rh, cws[5], rh, cont['gross_weight_kg'])
            self._annotate("measurement",      M + sum(cws[:6]), rty - rh, cws[6], rh, cont['measurement'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (label, val) in enumerate([
            ("FREIGHT TERMS", d['freight_terms']),
            ("ORIGINALS ISSUED", d['number_of_originals']),
            ("SERVICE", d['service_type']),
        ]):
            lbox(M + i * thirds, y, thirds, 1.4*cm, label, val)

        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W / 2, 0.28*cm,
            "CMA CGM S.A. | 4, Quai d'Arenc, 13002 Marseille, France | www.cma-cgm.com")


# ─────────────────────────────────────────────
#  Hapag-Lloyd Renderer
# ─────────────────────────────────────────────

class HapagLloydRenderer(BLRenderer):
    PRIMARY   = colors.HexColor("#E05206")
    SECONDARY = colors.HexColor("#333333")
    BLUE      = colors.HexColor("#003399")
    BORDER    = colors.HexColor("#BBBBBB")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # En-tête
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 0.5*cm, W, 0.5*cm, fill=1, stroke=0)
        c.setFillColor(self.SECONDARY)
        c.rect(0, H - 2.8*cm, W, 2.3*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 19)
        c.drawString(M, H - 1.9*cm, "Hapag-Lloyd")
        c.setFont("Helvetica", 7.5)
        c.drawString(M, H - 2.35*cm, "HAPAG-LLOYD AKTIENGESELLSCHAFT · BALLINDAMM 25 · D-20095 HAMBURG")
        c.setFont("Helvetica-Bold", 13)
        c.drawRightString(W - M, H - 1.7*cm, "BILL OF LADING")
        c.setFont("Helvetica", 7.5)
        c.drawRightString(W - M, H - 2.1*cm, "FOR COMBINED TRANSPORT OR PORT-TO-PORT SHIPMENT")
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 3.1*cm, W, 0.3*cm, fill=1, stroke=0)

        # Info BL
        c.setFillColor(colors.HexColor("#F0F0F0"))
        c.rect(0, H - 3.6*cm, W, 0.5*cm, fill=1, stroke=0)
        c.setFillColor(self.SECONDARY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 3.35*cm, f"B/L NO.: {d['bl_number']}")
        c.setFont("Helvetica", 8)
        c.drawString(W / 2 - 2*cm, H - 3.35*cm, f"ISSUE DATE: {d['issue_date']}")
        c.drawRightString(W - M, H - 3.35*cm, f"PLACE OF ISSUE: {d['issue_place']}")
        self._annotate("bl_number", M, H - 3.6*cm, 8*cm, 0.5*cm, d['bl_number'])

        top = H - 3.9*cm
        c.setStrokeColor(self.BORDER)
        label_h = 0.35 * cm

        def hl_box(x, y, w, h, label, val, multiline=False, fs=7.5):
            field_name = label.lower().replace(" ", "_")
            content_top = y - label_h
            content_h = h - label_h
            c.setStrokeColor(self.BORDER)
            c.setFillColor(colors.HexColor("#F8F8F8"))
            c.rect(x, y - h, w, h, fill=1, stroke=1)
            c.setFillColor(self.BLUE)
            c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - 0.23*cm, label)
            c.setFillColor(colors.black)
            if multiline:
                self._draw_multiline_field(field_name,
                    x, content_top, w, content_h, val, size=fs)
            else:
                self._draw_text_field(field_name,
                    x, content_top, w, content_h, val, size=fs)

        half = (W - 2*M) / 2
        rh = 2.8*cm

        hl_box(M, top, half, rh, "SHIPPER", d['shipper']['full'], multiline=True)
        hl_box(M + half, top, half, rh, "CONSIGNEE", d['consignee']['full'], multiline=True)

        y = top - rh
        hl_box(M, y, half, rh, "NOTIFY PARTY", d['notify_party']['full'], multiline=True)

        quarter = half / 2
        hl_box(M + half, y, quarter, rh / 2, "VESSEL", d['vessel'])
        hl_box(M + half + quarter, y, quarter, rh / 2, "VOYAGE", d['voyage'])
        hl_box(M + half, y - rh / 2, quarter, rh / 2, "PORT OF LOADING", d['port_of_loading'], fs=6.5)
        hl_box(M + half + quarter, y - rh / 2, quarter, rh / 2, "PORT OF DISCHARGE", d['port_of_discharge'], fs=6.5)

        y -= rh
        q4 = (W - 2*M) / 4
        for i, (label, val) in enumerate([
            ("PLACE OF RECEIPT", d['place_of_receipt']),
            ("PLACE OF DELIVERY", d['place_of_delivery']),
            ("ETD", d['etd']),
            ("ETA", d['eta']),
        ]):
            hl_box(M + i * q4, y, q4, 1.2*cm, label, val, fs=6.5)

        y -= 1.2*cm + 0.3*cm
        headers = ["CONTAINER NO.", "SEAL NO.", "TYPE", "PKGS", "DESCRIPTION", "GROSS WT (KG)", "CBM"]
        cws = [3.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 4.8*cm, 2.8*cm, 2.6*cm]
        total_w = W - 2*M
        rows = [
            [c['container_number'], c['seal_number'], c['container_type'],
             c['packages']['description'], c['commodity'],
             f"{c['gross_weight_kg']} KG", c['measurement']]
            for c in d['containers']
        ]
        bottom_y, row_top_ys, rh = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows,
            header_bg=self.SECONDARY, border_color=self.BORDER)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,             rty - rh, cws[0], rh, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],    rty - rh, cws[1], rh, cont['seal_number'])
            self._annotate("gross_weight_kg",  M + sum(cws[:5]), rty - rh, cws[5], rh, cont['gross_weight_kg'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (label, val) in enumerate([
            ("FREIGHT PAYABLE AT", d['freight_terms']),
            ("NO. OF ORIGINAL B/L", d['number_of_originals']),
            ("DATE OF ISSUE", d['issue_date']),
        ]):
            hl_box(M + i * thirds, y, thirds, 1.4*cm, label, val)

        c.setFillColor(self.SECONDARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W / 2, 0.28*cm,
            "Hapag-Lloyd AG | Ballindamm 25 | D-20095 Hamburg, Germany | www.hapag-lloyd.com")


# ─────────────────────────────────────────────
#  ONE (Ocean Network Express) Renderer
# ─────────────────────────────────────────────

class ONERenderer(BLRenderer):
    PRIMARY   = colors.HexColor("#E84393")   # ONE magenta
    SECONDARY = colors.HexColor("#1C1C1C")
    BG_LIGHT  = colors.HexColor("#FFF0F7")
    BORDER    = colors.HexColor("#F0A0CC")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Left magenta sidebar + dark header
        c.setFillColor(self.SECONDARY)
        c.rect(0, H - 3.0*cm, W, 3.0*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 3.0*cm, 0.7*cm, 3.0*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(1.1*cm, H - 1.6*cm, "ONE")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#CCCCCC"))
        c.drawString(1.1*cm, H - 2.1*cm, "OCEAN NETWORK EXPRESS PTE. LTD.")
        c.drawString(1.1*cm, H - 2.6*cm, "7 STRAITS VIEW, MARINA ONE, SINGAPORE 018936")
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(W - M, H - 1.6*cm, "BILL OF LADING")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(self.PRIMARY)
        c.drawRightString(W - M, H - 2.1*cm, "ORIGINAL NON-NEGOTIABLE")

        # BL info band
        c.setFillColor(self.BG_LIGHT)
        c.rect(0, H - 3.55*cm, W, 0.55*cm, fill=1, stroke=0)
        c.setFillColor(self.SECONDARY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 3.3*cm, f"B/L NO.: {d['bl_number']}")
        c.setFont("Helvetica", 8)
        c.drawString(W/2 - 2*cm, H - 3.3*cm, f"DATE: {d['issue_date']}")
        c.drawRightString(W - M, H - 3.3*cm, f"PLACE: {d['issue_place']}")
        self._annotate("bl_number", M, H - 3.55*cm, 8*cm, 0.55*cm, d['bl_number'])

        top = H - 3.85*cm
        label_h = 0.38*cm

        def obox(x, y, w, h, label, val, multiline=False, fs=7.5):
            fn = label.lower().replace(" ", "_").replace("/", "_").rstrip("_")
            c.setFillColor(self.PRIMARY)
            c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - label_h + 3, label)
            c.setFillColor(self.BG_LIGHT)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h - label_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            if multiline:
                self._draw_multiline_field(fn, x, y - label_h, w, h - label_h, val, size=fs)
            else:
                self._draw_text_field(fn, x, y - label_h, w, h - label_h, val, size=fs)

        half = (W - 2*M) / 2
        rh = 2.8*cm
        obox(M, top, half, rh, "SHIPPER / EXPORTER", d['shipper']['full'], multiline=True)
        obox(M + half, top, half, rh, "CONSIGNEE", d['consignee']['full'], multiline=True)

        y = top - rh
        obox(M, y, half, rh, "NOTIFY PARTY", d['notify_party']['full'], multiline=True)
        quarter = half / 2
        obox(M + half, y, quarter, rh/2, "VESSEL", d['vessel'])
        obox(M + half + quarter, y, quarter, rh/2, "VOYAGE NO.", d['voyage'])
        obox(M + half, y - rh/2, quarter, rh/2, "PORT OF LOADING", d['port_of_loading'], fs=6.5)
        obox(M + half + quarter, y - rh/2, quarter, rh/2, "PORT OF DISCHARGE", d['port_of_discharge'], fs=6.5)

        y -= rh
        q4 = (W - 2*M) / 4
        for i, (lbl, val) in enumerate([
            ("PLACE OF RECEIPT", d['place_of_receipt']),
            ("PLACE OF DELIVERY", d['place_of_delivery']),
            ("ETD", d['etd']),
            ("ETA", d['eta']),
        ]):
            obox(M + i*q4, y, q4, 1.2*cm, lbl, val, fs=6.5)

        y -= 1.2*cm + 0.3*cm
        headers = ["CONTAINER NO.", "SEAL NO.", "TYPE", "PACKAGES", "DESCRIPTION", "GROSS WT (KG)", "CBM"]
        cws     = [3.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 4.8*cm, 2.8*cm, 2.6*cm]
        total_w = W - 2*M
        rows = [
            [c2['container_number'], c2['seal_number'], c2['container_type'],
             c2['packages']['description'], c2['commodity'],
             f"{c2['gross_weight_kg']} KG", c2['measurement']]
            for c2 in d['containers']
        ]
        bottom_y, row_top_ys, rh2 = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows, header_bg=self.PRIMARY,
            border_color=self.BORDER)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,              rty - rh2, cws[0], rh2, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],     rty - rh2, cws[1], rh2, cont['seal_number'])
            self._annotate("container_type",   M + sum(cws[:2]), rty - rh2, cws[2], rh2, cont['container_type'])
            self._annotate("gross_weight_kg",  M + sum(cws[:5]), rty - rh2, cws[5], rh2, cont['gross_weight_kg'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (lbl, val) in enumerate([
            ("FREIGHT TERMS", d['freight_terms']),
            ("NO. OF ORIGINALS", d['number_of_originals']),
            ("SERVICE TYPE", d['service_type']),
        ]):
            obox(M + i*thirds, y, thirds, 1.4*cm, lbl, val)

        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.3*cm,
            "Ocean Network Express Pte. Ltd. | Marina One, Singapore | www.one-line.com")


# ─────────────────────────────────────────────
#  Evergreen Renderer
# ─────────────────────────────────────────────

class EvergreenRenderer(BLRenderer):
    PRIMARY   = colors.HexColor("#009444")   # Evergreen green
    SECONDARY = colors.HexColor("#004d23")   # Dark green
    BG_LIGHT  = colors.HexColor("#E8F5E9")
    BORDER    = colors.HexColor("#81C784")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Green top band
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.6*cm, W, 2.6*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(M, H - 1.6*cm, "EVERGREEN")
        c.setFont("Helvetica", 7.5)
        c.drawString(M, H - 2.1*cm, "EVERGREEN MARINE CORPORATION (TAIWAN) LTD.")
        c.setFont("Helvetica-Bold", 14)
        c.drawRightString(W - M, H - 1.5*cm, "BILL OF LADING")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#CCFFCC"))
        c.drawRightString(W - M, H - 2.05*cm, "COMBINED TRANSPORT")

        # Dark green info strip
        c.setFillColor(self.SECONDARY)
        c.rect(0, H - 3.15*cm, W, 0.55*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 2.9*cm, f"B/L No.: {d['bl_number']}")
        c.setFont("Helvetica", 8)
        c.drawString(W/2 - 2*cm, H - 2.9*cm, f"DATE: {d['issue_date']}")
        c.drawRightString(W - M, H - 2.9*cm, f"PLACE: {d['issue_place']}")
        self._annotate("bl_number", M, H - 3.15*cm, 8*cm, 0.55*cm, d['bl_number'])

        top = H - 3.45*cm
        label_h = 0.36*cm

        def gbox(x, y, w, h, label, val, multiline=False, fs=7.5):
            fn = label.lower().replace(" ", "_").replace("(", "").replace(")", "").strip("_")
            c.setFillColor(self.SECONDARY)
            c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - label_h + 3, label)
            c.setFillColor(self.BG_LIGHT)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h - label_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            if multiline:
                self._draw_multiline_field(fn, x, y - label_h, w, h - label_h, val, size=fs)
            else:
                self._draw_text_field(fn, x, y - label_h, w, h - label_h, val, size=fs)

        half = (W - 2*M) / 2
        rh = 2.8*cm
        gbox(M, top, half, rh, "SHIPPER / EXPORTER", d['shipper']['full'], multiline=True)
        gbox(M + half, top, half, rh, "CONSIGNEE", d['consignee']['full'], multiline=True)

        y = top - rh
        gbox(M, y, half, rh, "NOTIFY PARTY", d['notify_party']['full'], multiline=True)
        quarter = half / 2
        gbox(M + half, y, quarter, rh/2, "VESSEL", d['vessel'])
        gbox(M + half + quarter, y, quarter, rh/2, "VOYAGE NO.", d['voyage'])
        gbox(M + half, y - rh/2, quarter, rh/2, "PORT OF LOADING", d['port_of_loading'], fs=6.5)
        gbox(M + half + quarter, y - rh/2, quarter, rh/2, "PORT OF DISCHARGE", d['port_of_discharge'], fs=6.5)

        y -= rh
        q4 = (W - 2*M) / 4
        for i, (lbl, val) in enumerate([
            ("PLACE OF RECEIPT", d['place_of_receipt']),
            ("PLACE OF DELIVERY", d['place_of_delivery']),
            ("ETD", d['etd']),
            ("ETA", d['eta']),
        ]):
            gbox(M + i*q4, y, q4, 1.2*cm, lbl, val, fs=6.5)

        y -= 1.2*cm + 0.3*cm
        headers = ["CONTAINER NO.", "SEAL NO.", "TYPE", "PACKAGES", "DESCRIPTION", "GROSS WT (KG)", "CBM"]
        cws     = [3.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 4.8*cm, 2.8*cm, 2.6*cm]
        total_w = W - 2*M
        rows = [
            [c2['container_number'], c2['seal_number'], c2['container_type'],
             c2['packages']['description'], c2['commodity'],
             f"{c2['gross_weight_kg']} KG", c2['measurement']]
            for c2 in d['containers']
        ]
        bottom_y, row_top_ys, rh2 = self._draw_multi_cargo_table(
            M, y, total_w, cws, headers, rows,
            header_bg=self.PRIMARY, border_color=self.BORDER)
        for cont, rty in zip(d['containers'], row_top_ys):
            self._annotate("container_number", M,              rty - rh2, cws[0], rh2, cont['container_number'])
            self._annotate("seal_number",      M + cws[0],     rty - rh2, cws[1], rh2, cont['seal_number'])
            self._annotate("container_type",   M + sum(cws[:2]), rty - rh2, cws[2], rh2, cont['container_type'])
            self._annotate("gross_weight_kg",  M + sum(cws[:5]), rty - rh2, cws[5], rh2, cont['gross_weight_kg'])

        y = bottom_y - 0.3*cm
        thirds = total_w / 3
        for i, (lbl, val) in enumerate([
            ("FREIGHT PAYABLE AT", d['freight_terms']),
            ("NO. OF ORIGINALS", d['number_of_originals']),
            ("SERVICE TYPE", d['service_type']),
        ]):
            gbox(M + i*thirds, y, thirds, 1.4*cm, lbl, val)

        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.3*cm,
            "Evergreen Marine Corporation (Taiwan) Ltd. | Taipei, Taiwan | www.evergreen-marine.com")


# ─────────────────────────────────────────────
#  Registre
# ─────────────────────────────────────────────

RENDERERS = {
    "msc":        MSCRenderer,
    "maersk":     MaerskRenderer,
    "cmacgm":     CMACGMRenderer,
    "hapaglloyd": HapagLloydRenderer,
    "one":        ONERenderer,
    "evergreen":  EvergreenRenderer,
}


# ─────────────────────────────────────────────
#  Génération du dataset
# ─────────────────────────────────────────────

def generate_dataset(count: int, output_dir: str, formats: list, seed: int = 42):
    random.seed(seed)
    Faker.seed(seed)

    pdf_dir = Path(output_dir) / "pdfs"
    ann_dir = Path(output_dir) / "annotations"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    dataset_annotations = []
    counters = {f: 0 for f in formats}

    print(f"\n{'='*55}")
    print(f"  Bill of Lading Synthetic Dataset Generator")
    print(f"  (Version corrigée — corrections appliquées)")
    print(f"{'='*55}")
    print(f"  Count    : {count}")
    print(f"  Formats  : {', '.join(formats)}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*55}\n")

    for i in range(count):
        carrier_key = formats[i % len(formats)]
        counters[carrier_key] += 1

        data = build_bl_data(carrier_key)
        bl_num = data['bl_number']
        filename = f"bl_{carrier_key}_{i+1:04d}_{bl_num}.pdf"
        pdf_path = str(pdf_dir / filename)

        renderer_cls = RENDERERS[carrier_key]
        renderer = renderer_cls(pdf_path, data)
        renderer.render()
        annotations = renderer.save()

        record = {
            "id": i + 1,
            "filename": filename,
            "carrier": carrier_key,
            "bl_number": bl_num,
            "n_containers": data['n_containers'],
            "fields": {
                "bl_number": bl_num,
                "shipper_name": data['shipper']['name'],
                "consignee_name": data['consignee']['name'],
                "notify_party": data['notify_party']['name'],
                "vessel": data['vessel'],
                "voyage": data['voyage'],
                "port_of_loading": data['port_of_loading'],
                "port_of_discharge": data['port_of_discharge'],
                "place_of_receipt": data['place_of_receipt'],
                "place_of_delivery": data['place_of_delivery'],
                "etd": data['etd'],
                "eta": data['eta'],
                # First container fields (backward compat for invoice/packing_list generators)
                "container_number": data['container_number'],
                "container_type": data['container_type'],
                "seal_number": data['seal_number'],
                "commodity": data['commodity'],
                "packages": data['packages']['description'],
                "gross_weight_kg": data['gross_weight_kg'],
                "measurement": data['measurement'],
                # All containers (used by ConsistencyEngine)
                "containers": data['containers'],
                "freight_terms": data['freight_terms'],
                "issue_date": data['issue_date'],
                "issue_place": data['issue_place'],
                "number_of_originals": data['number_of_originals'],
                "service_type": data['service_type'],
            },
            "bbox_annotations": annotations,
        }
        dataset_annotations.append(record)

        ann_path = ann_dir / filename.replace(".pdf", ".json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if (i + 1) % 50 == 0 or i == count - 1:
            print(f"  [{i+1:4d}/{count}] Generated: {filename}")

    dataset = {
        "metadata": {
            "total_documents": count,
            "carriers": formats,
            "distribution": counters,
            "generation_date": datetime.now().isoformat(),
            "fixes_applied": [
                "ETA always >= ETD + transit time",
                "ASCII-only addresses (en_US Faker)",
                "Cargo table column separators drawn",
                "TYPE and PKGS in separate columns",
                "place_of_receipt/delivery coherent with POL/POD",
                "BL number annotated in header band",
            ],
            "bbox_format": "[x, y, width, height]",
            "coordinate_system": "PDF points, origin bottom-left",
        },
        "documents": dataset_annotations,
    }

    dataset_path = Path(output_dir) / "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    _write_splits(dataset_annotations, output_dir)

    print(f"\n{'='*55}")
    print(f"  [OK] {count} B/L generated successfully")
    print(f"  [OK] Distribution: {counters}")
    print(f"  [OK] Dataset JSON: {dataset_path}")
    print(f"{'='*55}\n")
    return dataset


def _write_splits(records: list, output_dir: str):
    shuffled = records[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    splits = {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }
    for split_name, split_records in splits.items():
        split_path = Path(output_dir) / f"{split_name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": split_name,
                "count": len(split_records),
                "documents": split_records
            }, f, ensure_ascii=False, indent=2)
        print(f"  [OK] {split_name:5s} split: {len(split_records):3d} documents -> {split_path.name}")


# ─────────────────────────────────────────────
#  Génération paramétrique (22 layouts × N docs)
# ─────────────────────────────────────────────

def generate_dataset_parametric(count_per_layout: int, output_dir: str, seed: int = 0):
    """
    Génère count_per_layout documents pour chacun des 22 layouts BL.
    Total : count_per_layout × 22 documents.
    """
    from layout_engine import BL_LAYOUTS
    from layout_engine.renderer_bl import BLParametricRenderer

    if seed:
        random.seed(seed)
        Faker.seed(seed)

    pdf_dir = Path(output_dir) / "pdfs"
    ann_dir = Path(output_dir) / "annotations"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    total   = count_per_layout * len(BL_LAYOUTS)
    records = []
    doc_idx = 0

    print(f"\n{'='*60}")
    print(f"  BL Parametric Generator — {len(BL_LAYOUTS)} layouts × {count_per_layout} docs = {total}")
    print(f"{'='*60}\n")

    for cfg in BL_LAYOUTS:
        layout_count = 0
        while layout_count < count_per_layout:
            doc_idx += 1
            layout_count += 1

            data     = build_bl_data(cfg.layout_id.split("-")[0].lower() if False else "msc")
            bl_num   = data["bl_number"]
            filename = f"bl_{cfg.layout_id.lower()}_{doc_idx:05d}_{bl_num}.pdf"
            pdf_path = str(pdf_dir / filename)

            renderer = BLParametricRenderer(pdf_path, data, cfg)
            renderer.render()
            annotations = renderer.save()

            record = {
                "id":          doc_idx,
                "filename":    filename,
                "layout_id":   cfg.layout_id,
                "bl_number":   bl_num,
                "n_containers":data["n_containers"],
                "fields": {
                    "bl_number":            bl_num,
                    "shipper_name":         data["shipper"]["name"],
                    "shipper_address":      data["shipper"]["address"],
                    "consignee_name":       data["consignee"]["name"],
                    "consignee_address":    data["consignee"]["address"],
                    "notify_party_name":    data["notify_party"]["name"],
                    "notify_party_address": data["notify_party"]["address"],
                    "vessel":               data["vessel"],
                    "voyage":               data["voyage"],
                    "port_of_loading":      data["port_of_loading"],
                    "port_of_discharge":    data["port_of_discharge"],
                    "place_of_receipt":     data["place_of_receipt"],
                    "place_of_delivery":    data["place_of_delivery"],
                    "etd":                  data["etd"],
                    "eta":                  data["eta"],
                    "container_number":     data["container_number"],
                    "container_type":       data["container_type"],
                    "seal_number":          data["seal_number"],
                    "commodity":            data["commodity"],
                    "packages":             data["packages"]["description"],
                    "gross_weight_kg":      data["gross_weight_kg"],
                    "measurement":          data["measurement"],
                    "containers":           data["containers"],
                    "freight_terms":        data["freight_terms"],
                    "issue_date":           data["issue_date"],
                    "issue_place":          data["issue_place"],
                    "number_of_originals":  data["number_of_originals"],
                    "service_type":         data["service_type"],
                },
                "bbox_annotations": annotations,
            }
            records.append(record)

            ann_path = ann_dir / filename.replace(".pdf", ".json")
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"  [{cfg.layout_id}] {layout_count} docs générés")

    dataset = {
        "metadata": {
            "total_documents": len(records),
            "generator":       "parametric_v2",
            "layouts":         [c.layout_id for c in BL_LAYOUTS],
            "count_per_layout":count_per_layout,
            "generation_date": datetime.now().isoformat(),
        },
        "documents": records,
    }
    dataset_path = Path(output_dir) / "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    _write_splits(records, output_dir)

    print(f"\n{'='*60}")
    print(f"  [OK] {len(records)} BL générés → {dataset_path}")
    print(f"{'='*60}\n")
    return dataset


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Bill of Lading dataset")
    parser.add_argument("--count",            type=int,  default=400,
                        help="Nombre total de docs (mode classique)")
    parser.add_argument("--count-per-layout", type=int,  default=50,
                        help="Docs par layout (mode --parametric)")
    parser.add_argument("--output",           type=str,  default="./bl_dataset")
    parser.add_argument("--formats",          type=str,  default="all",
                        help="all | msc,maersk,... (mode classique uniquement)")
    parser.add_argument("--seed",             type=int,  default=0)
    parser.add_argument("--parametric",       action="store_true",
                        help="Utiliser les 22 layouts paramétriques (recommandé)")
    args = parser.parse_args()

    if args.parametric:
        generate_dataset_parametric(args.count_per_layout, args.output, args.seed)
    else:
        if args.formats == "all":
            selected_formats = list(RENDERERS.keys())
        else:
            selected_formats = [f.strip().lower() for f in args.formats.split(",")]
            for f in selected_formats:
                if f not in RENDERERS:
                    raise ValueError(f"Format inconnu '{f}'. Choisir parmi: {list(RENDERERS.keys())}")
        generate_dataset(args.count, args.output, selected_formats, args.seed)
