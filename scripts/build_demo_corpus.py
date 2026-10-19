"""
Build the bundled demo corpus (demo_data/*.pdf).

The demo PDFs are generated from the text in this file so their content
is reviewable and reproducible. All content is illustrative: it was
written for demonstrating the FinTech Consumer Insight Engine, contains
no survey data or statistics, and every page says so.

Usage (from the project root):

    python scripts/build_demo_corpus.py

Uses no dependencies beyond the standard library: it writes simple
text-only PDFs (Helvetica, WinAnsi encoding) that pypdf can extract.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo_data"

DISCLAIMER = (
    "Illustrative demo content written for the FinTech Consumer Insight Engine. "
    "Not empirical research: no survey was conducted and no real respondent data is reported."
)

# Each document: (file name, title, [pages]); each page: [(kind, text)],
# where kind is "h" (heading) or "p" (paragraph).
DOCUMENTS = [
    (
        "digital_payments_adoption_demo_report.pdf",
        "Digital Payments Adoption — Demo Research Report",
        [
            [
                ("h", "About this document"),
                ("p", "This is an illustrative demo report about how consumers adopt digital payments such as mobile "
                      "wallets, contactless cards, and app-based transfers. It was written to demonstrate semantic "
                      "search, cited question answering, consumer insights, and report generation. The findings below "
                      "are example hypotheses of the kind a research team might investigate, not measured results."),
                ("h", "Executive summary"),
                ("p", "In this illustrative scenario, adoption of digital payments is driven mainly by convenience, "
                      "speed at checkout, and integration with apps consumers already use. The main barriers are "
                      "unclear or hidden fees, gaps in merchant acceptance, low confidence with new technology, and "
                      "concerns about security. Different consumer segments weigh these factors differently, so a "
                      "single message rarely works for every group."),
            ],
            [
                ("h", "Adoption drivers"),
                ("p", "Convenience: example interview themes describe paying with a phone as faster than finding a "
                      "card or counting cash, especially for small everyday purchases."),
                ("p", "Speed at checkout: tap-to-pay and saved payment details reduce friction in queues and in online "
                      "checkout flows, which consumers in this scenario describe as a reason to keep using a wallet."),
                ("p", "Integration: wallets built into phones, messaging apps, and transit or retail apps lower the "
                      "effort of getting started because no separate sign-up is needed."),
                ("p", "Rewards and perks: loyalty points and cashback offers are described as an initial incentive to "
                      "try a new payment method, though they are less often cited as a reason to keep using it."),
                ("p", "Social influence: seeing friends and family split bills or send money digitally is an example "
                      "of how peer behaviour can encourage first-time use."),
            ],
            [
                ("h", "Adoption barriers"),
                ("p", "Fees and pricing transparency: consumers in this scenario are wary of transfer fees, currency "
                      "conversion charges, and charges that only appear after a payment is made. Hidden or unclear fees "
                      "are an example of a barrier that damages trust as well as slowing adoption."),
                ("p", "Merchant acceptance: if a wallet is not accepted at shops a person visits often, they keep "
                      "carrying cards or cash and the wallet becomes an occasional extra rather than a habit."),
                ("p", "Digital confidence: some consumers are unsure how to set up a wallet, link an account, or "
                      "recover access after changing phones, and prefer familiar payment methods instead."),
                ("p", "Connectivity and device access: payment methods that depend on a recent smartphone or a "
                      "reliable data connection can exclude people with older devices or patchy coverage."),
                ("p", "Security worries: fear of fraud and of losing a phone are described as reasons to delay "
                      "adoption; these overlap with the trust themes covered in the companion demo report."),
            ],
            [
                ("h", "Consumer segments (illustrative)"),
                ("p", "Younger consumers: in this example they are described as early adopters who value speed and "
                      "peer-to-peer transfers, and who switch providers readily if an app feels slow or outdated."),
                ("p", "Older consumers: described as more cautious, valuing clear instructions, the option to talk to "
                      "a person, and reassurance about what happens if a payment goes wrong."),
                ("p", "Small business owners: interested in lower acceptance costs and faster settlement, but "
                      "concerned about equipment, setup effort, and transaction fees eating into margins."),
                ("p", "Underbanked consumers: may benefit most from low-cost digital accounts, but face barriers "
                      "around identity checks, device access, and fees on small balances."),
                ("p", "These segment descriptions are illustrative personas for demonstration, not results from a "
                      "segmented survey."),
            ],
            [
                ("h", "Transaction fees and payment method choice"),
                ("p", "In this scenario, consumers compare payment methods partly on cost, but the clarity of fees "
                      "matters as much as their size. A small fee that is shown up front is described as more "
                      "acceptable than an unexpected charge discovered later on a statement."),
                ("p", "Example implication: providers that display the full cost before a payment is confirmed, "
                      "including any conversion charges, may reduce abandonment and complaints."),
                ("h", "Example recommendations"),
                ("p", "These are suggestions derived from the illustrative themes above, not validated findings: "
                      "make fees visible before confirmation; expand merchant acceptance in everyday categories; "
                      "offer guided setup and in-app help for first-time users; and design for older devices and "
                      "low-connectivity conditions."),
                ("h", "Method note"),
                ("p", "No data was collected for this document. It was written to give the demo realistic, "
                      "well-structured material to search and cite."),
            ],
        ],
    ),
    (
        "consumer_trust_digital_payments_demo_report.pdf",
        "Consumer Trust in Digital Payments — Demo Research Report",
        [
            [
                ("h", "About this document"),
                ("p", "This is an illustrative demo report about the role of trust in consumers' willingness to use "
                      "digital payments. It contains example themes and hypotheses written for demonstrating the "
                      "FinTech Consumer Insight Engine. It does not report survey results or statistics."),
                ("h", "Executive summary"),
                ("p", "In this illustrative scenario, trust is built through visible security, clear communication "
                      "about data use, a fair and fast way to resolve problems, and the reputation of the provider. "
                      "Trust is lost quickly after an unexpected fee, an unexplained declined payment, or a slow "
                      "response to suspected fraud."),
            ],
            [
                ("h", "What builds trust"),
                ("p", "Visible security features: consumers in this example describe biometric login, transaction "
                      "alerts, and the ability to freeze a card instantly as signs that a provider takes security "
                      "seriously."),
                ("p", "Provider reputation: established banks and well-known brands are described as starting with "
                      "more trust, while newer providers need to earn it through transparency and reliability."),
                ("p", "Regulation and protections: knowing that a provider is regulated and that payments are "
                      "protected is an example of reassurance that matters most to cautious users."),
                ("p", "Reliability: payments that work every time, without unexplained declines, are described as "
                      "the quiet foundation of trust."),
            ],
            [
                ("h", "Privacy and data sharing concerns"),
                ("p", "Consumers in this scenario are uneasy about linking bank accounts to new apps and about how "
                      "their transaction data might be shared with third parties or used for marketing."),
                ("p", "Example theme: plain-language explanations of what data is collected, why it is needed, and "
                      "how to opt out are described as more reassuring than long legal privacy policies."),
                ("p", "Example theme: asking for only the permissions a feature needs, at the moment it needs them, "
                      "is described as a signal that the provider respects the user."),
            ],
            [
                ("h", "Fraud and dispute resolution"),
                ("p", "In this illustrative scenario, fear of fraud is less about whether fraud happens and more "
                      "about what happens next: how quickly money is returned, whether a real person helps, and "
                      "whether the process is clear."),
                ("p", "A fast, well-explained dispute process is described as turning a bad experience into a reason "
                      "to trust a provider more, while a slow or opaque process drives people back to cash or cards."),
                ("h", "Transparency, fees, and communication"),
                ("p", "Unexpected fees are an example of a trust breaker as well as a cost issue. Consumers in this "
                      "scenario expect to see fees and exchange rates before they confirm a payment, and to receive "
                      "clear notifications when something changes."),
            ],
            [
                ("h", "Differences across consumer groups (illustrative)"),
                ("p", "Younger consumers are described as trusting app-based providers readily but abandoning them "
                      "quickly after a poor experience. Older consumers are described as slower to trust but more "
                      "loyal once trust is established, and as valuing human support."),
                ("p", "These differences are illustrative personas for demonstration, not measured group results."),
                ("h", "Example recommendations"),
                ("p", "Suggestions derived from the illustrative themes, not validated findings: make security "
                      "features visible during onboarding; explain data use in plain language; show all fees before "
                      "confirmation; and offer a fast dispute process with access to a person."),
                ("h", "Method note"),
                ("p", "No data was collected for this document. It was written as realistic demo material."),
            ],
        ],
    ),
]

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
MARGIN_X, TOP_Y, BOTTOM_Y = 60, 740, 70
BODY_SIZE, HEADING_SIZE, TITLE_SIZE, FOOTER_SIZE = 10.5, 13, 16, 8
LEADING = 14.5
WRAP_BODY, WRAP_FOOTER = 92, 118


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _page_stream(title: str, page_no: int, page_count: int, blocks: list[tuple[str, str]]) -> bytes:
    ops: list[bytes] = []
    y = TOP_Y

    def line(text: str, font: str, size: float, x: float = MARGIN_X) -> None:
        nonlocal y
        ops.append(b"BT /%s %.1f Tf %.1f %.1f Td (" % (font.encode(), size, x, y) + _escape(text) + b") Tj ET")

    line(title, "F2", TITLE_SIZE)
    y -= LEADING * 1.6
    for kind, text in blocks:
        if kind == "h":
            y -= LEADING * 0.4
            line(text, "F2", HEADING_SIZE)
            y -= LEADING * 1.3
        else:
            for wrapped in textwrap.wrap(text, WRAP_BODY):
                line(wrapped, "F1", BODY_SIZE)
                y -= LEADING
            y -= LEADING * 0.5
        if y < BOTTOM_Y + LEADING:
            raise ValueError(f"Page {page_no} of '{title}' overflows; split it into more pages.")

    footer_y = 40
    for i, footer_line in enumerate(textwrap.wrap(DISCLAIMER, WRAP_FOOTER)):
        ops.append(b"BT /F1 %.1f Tf %.1f %.1f Td (" % (FOOTER_SIZE, MARGIN_X, footer_y + 10 - i * 10) + _escape(footer_line) + b") Tj ET")
    ops.append(b"BT /F1 %.1f Tf %.1f %.1f Td (" % (FOOTER_SIZE, PAGE_WIDTH - MARGIN_X - 40, 20) + _escape(f"Page {page_no} of {page_count}") + b") Tj ET")
    return b"\n".join(ops)


def build_pdf(path: Path, title: str, pages: list[list[tuple[str, str]]]) -> None:
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        # Info strings use PDFDocEncoding, not WinAnsi; UTF-16BE (with BOM) keeps the em dash intact.
        5: b"<< /Title <" + ("﻿" + title).encode("utf-16-be").hex().upper().encode()
        + b"> /Subject (Illustrative demo content) /Producer (build_demo_corpus.py) >>",
    }
    page_ids = []
    for index, blocks in enumerate(pages):
        page_id, content_id = 6 + index * 2, 7 + index * 2
        page_ids.append(page_id)
        stream = _page_stream(title, index + 1, len(pages), blocks)
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] " % (PAGE_WIDTH, PAGE_HEIGHT)
            + b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % content_id
        )
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objects[2] = b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids) + b"] /Count %d >>" % len(page_ids)

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b"%d 0 obj\n" % obj_id + objects[obj_id] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offsets[i] for i in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at)
    path.write_bytes(bytes(out))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for file_name, title, pages in DOCUMENTS:
        build_pdf(OUTPUT_DIR / file_name, title, pages)
        print(f"Wrote {OUTPUT_DIR / file_name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
