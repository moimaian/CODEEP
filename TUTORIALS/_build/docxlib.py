# -*- coding: utf-8 -*-
"""Pequena camada sobre python-docx usada para montar o manual do CODEEP.

Marcação inline aceita nos textos:  **negrito**  e  `monoespaçado`.
"""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

COR = RGBColor(0xC8, 0x48, 0x48)       # coral do logotipo
COR_HEX = "C84848"
CINZA = RGBColor(0x59, 0x59, 0x59)
FONTE = "Calibri"
FONTE_MONO = "Consolas"
LARGURA_UTIL_CM = 16.0


def _rfonts(el, nome):
    rpr = el.get_or_add_rPr() if hasattr(el, "get_or_add_rPr") else el
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), nome)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


def _shade(p, fill):
    ppr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def _border_left(p, cor, sz=18):
    ppr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    l = OxmlElement("w:left")
    l.set(qn("w:val"), "single"); l.set(qn("w:sz"), str(sz))
    l.set(qn("w:space"), "6"); l.set(qn("w:color"), cor)
    b.append(l); ppr.append(b)


def _campo(par, instr, texto_cache="1"):
    """Insere um campo simples (PAGE etc.) com valor em cache."""
    def run_fld(tipo):
        r = par.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tipo)
        r._r.append(f); return r
    run_fld("begin")
    r = par.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "; r._r.append(it)
    run_fld("separate")
    par.add_run(texto_cache)
    run_fld("end")


TEXTOS = {
    "pt": {"rodape": "CODEEP — Manual do Usuário · versão {v}", "sumario": "Sumário",
           "figura": "Figura", "nota": "Observação:"},
    "en": {"rodape": "CODEEP — User Manual · version {v}", "sumario": "Contents",
           "figura": "Figure", "nota": "Note:"},
}


class Manual:
    def __init__(self, versao, paginas_toc=None, lang="pt"):
        self.lang = lang
        self.L = TEXTOS[lang]
        self.d = Document()
        self.versao = versao
        self.titulos = []          # (nivel, texto)
        self.paginas = paginas_toc or {}
        self.n_fig = 0
        self._config()

    # ---------------------------------------------------------------- estilos
    def _config(self):
        d = self.d
        sec = d.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.5)
        sec.top_margin = Cm(2.3); sec.bottom_margin = Cm(2.2)
        sec.different_first_page_header_footer = True

        st = d.styles["Normal"]
        st.font.name = FONTE; st.font.size = Pt(11)
        _rfonts(st.element, FONTE)
        pf = st.paragraph_format
        pf.space_after = Pt(6); pf.line_spacing = 1.15
        for nome, tam, antes, depois in (("Heading 1", 18, 20, 8), ("Heading 2", 14, 14, 6),
                                          ("Heading 3", 12, 10, 4)):
            h = d.styles[nome]
            h.font.name = FONTE; h.font.size = Pt(tam); h.font.bold = True
            h.font.italic = False; h.font.color.rgb = COR
            _rfonts(h.element, FONTE)
            h.paragraph_format.space_before = Pt(antes)
            h.paragraph_format.space_after = Pt(depois)
            h.paragraph_format.keep_with_next = True
        # rodapé
        f = sec.footer
        p = f.paragraphs[0]
        # o estilo Footer traz tabulações próprias (centro e direita para Letter); limpa-as
        tabs = OxmlElement("w:tabs")
        for pos in (4680, 9360):
            t = OxmlElement("w:tab"); t.set(qn("w:val"), "clear"); t.set(qn("w:pos"), str(pos)); tabs.append(t)
        p._p.get_or_add_pPr().append(tabs)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(LARGURA_UTIL_CM), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run(self.L["rodape"].format(v=self.versao) + "\t")
        r.font.size = Pt(9); r.font.color.rgb = CINZA
        _campo(p, "PAGE"); 
        for r in p.runs: r.font.size = Pt(9); r.font.color.rgb = CINZA

    # ---------------------------------------------------------------- inline
    def _inline(self, par, texto, tam=None, cor=None, italico=False):
        for parte in re.split(r"(\*\*.+?\*\*|`.+?`|(?<!\*)\*[^*\n]+?\*(?!\*))", texto):
            if not parte:
                continue
            if parte.startswith("**"):
                r = par.add_run(parte[2:-2]); r.bold = True
            elif parte.startswith("*") and parte.endswith("*") and len(parte) > 2:
                r = par.add_run(parte[1:-1]); r.italic = True
            elif parte.startswith("`"):
                r = par.add_run(parte[1:-1]); r.font.name = FONTE_MONO
                _rfonts(r._r, FONTE_MONO); r.font.size = Pt((tam or 11) - 1)
                r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
                continue
            else:
                r = par.add_run(parte)
            if tam: r.font.size = Pt(tam)
            if cor: r.font.color.rgb = cor
            if italico: r.italic = True

    # ---------------------------------------------------------------- blocos
    def h1(self, t):
        self.titulos.append((1, t)); return self.d.add_heading(t, level=1)

    def h2(self, t):
        self.titulos.append((2, t)); return self.d.add_heading(t, level=2)

    def h3(self, t):
        return self.d.add_heading(t, level=3)

    def p(self, t, alinhar=True):
        par = self.d.add_paragraph()
        if alinhar: par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._inline(par, t); return par

    def bullets(self, itens):
        for t in itens:
            par = self.d.add_paragraph(style="List Bullet")
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            par.paragraph_format.space_after = Pt(3)
            self._inline(par, t)
        self._espaco(3)

    def numerada(self, itens):
        for i, t in enumerate(itens, 1):
            par = self.d.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = par.paragraph_format
            pf.left_indent = Cm(0.9); pf.first_line_indent = Cm(-0.9); pf.space_after = Pt(3)
            pf.tab_stops.add_tab_stop(Cm(0.9))
            self._inline(par, f"{i}.\t{t}")
        self._espaco(3)

    def _espaco(self, pt):
        par = self.d.add_paragraph(); par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing = Pt(pt)
        for r in par.runs: r.font.size = Pt(1)

    def codigo(self, texto):
        par = self.d.add_paragraph()
        _shade(par, "F2F2F2")
        pf = par.paragraph_format
        pf.space_after = Pt(8); pf.line_spacing = 1.0
        pf.left_indent = Cm(0.2); pf.right_indent = Cm(0.2)
        linhas = texto.strip("\n").split("\n")
        for i, ln in enumerate(linhas):
            r = par.add_run(ln); r.font.name = FONTE_MONO; _rfonts(r._r, FONTE_MONO)
            r.font.size = Pt(9)
            if i < len(linhas) - 1: r.add_break()
        return par

    def nota(self, texto, rotulo=None):
        rotulo = rotulo or self.L["nota"]
        par = self.d.add_paragraph()
        _shade(par, "FBEDED"); _border_left(par, COR_HEX)
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = par.paragraph_format
        pf.left_indent = Cm(0.4); pf.space_before = Pt(4); pf.space_after = Pt(10)
        r = par.add_run(rotulo + " "); r.bold = True; r.font.size = Pt(10)
        self._inline(par, texto, tam=10)
        return par

    def figura(self, caminho, legenda, largura_cm=16.0):
        self.n_fig += 1
        par = self.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_before = Pt(6); par.paragraph_format.space_after = Pt(3)
        par.add_run().add_picture(caminho, width=Cm(largura_cm))
        cap = self.d.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        r = cap.add_run(f"{self.L['figura']} {self.n_fig} — {legenda}")
        r.italic = True; r.font.size = Pt(9); r.font.color.rgb = CINZA
        return self.n_fig

    def tabela(self, cabecalho, linhas, larguras_cm):
        t = self.d.add_table(rows=1, cols=len(cabecalho))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        def celula(c, texto, negrito=False, cor_txt=None, fill=None):
            c.text = ""
            par = c.paragraphs[0]; par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.space_before = Pt(2); par.paragraph_format.line_spacing = 1.0
            self._inline(par, texto, tam=9.5)
            for r in par.runs:
                if negrito: r.bold = True
                if cor_txt is not None: r.font.color.rgb = cor_txt
            if fill:
                tcpr = c._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
                tcpr.append(shd)
        for i, h in enumerate(cabecalho):
            celula(t.rows[0].cells[i], h, True, RGBColor(255, 255, 255), "404040")
        # repete o cabeçalho em cada página
        trpr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader")
        th.set(qn("w:val"), "true"); trpr.append(th)
        for k, lin in enumerate(linhas):
            cels = t.add_row().cells
            for i, v in enumerate(lin):
                celula(cels[i], v, fill=("F7F7F7" if k % 2 else None))
            cant = t.rows[-1]._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit")
            cs.set(qn("w:val"), "true"); cant.append(cs)
        tblpr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblpr.append(lay)
        tw = tblpr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW"); tblpr.append(tw)
        tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(int(sum(larguras_cm) * 567)))
        for i, w in enumerate(larguras_cm):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(larguras_cm):
                row.cells[i].width = Cm(w)
        if len(linhas) <= 6:   # tabelas curtas não devem ser partidas entre páginas
            for row in t.rows[:-1]:
                for c in row.cells:
                    for par in c.paragraphs: par.paragraph_format.keep_with_next = True
        self._espaco(8)
        return t

    def quebra(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------------------------------------------------------- capa/TOC
    def capa(self, logo, linhas):
        """linhas: lista de (texto, tamanho, negrito, cor, espaço_antes)"""
        par = self.d.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before = Pt(70)
        par.add_run().add_picture(logo, width=Cm(6.0))
        for texto, tam, neg, cor, antes in linhas:
            q = self.d.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_before = Pt(antes); q.paragraph_format.space_after = Pt(2)
            r = q.add_run(texto); r.font.size = Pt(tam); r.bold = neg
            if cor is not None: r.font.color.rgb = cor
        self.quebra()

    def sumario(self, estrutura):
        """Insere o campo TOC com o resultado já preenchido (estrutura = lista (nivel, texto))."""
        t = self.d.add_paragraph(); t.paragraph_format.space_after = Pt(10)
        r = t.add_run(self.L["sumario"]); r.bold = True; r.font.size = Pt(20); r.font.color.rgb = COR
        n = len(estrutura)
        for i, (nivel, texto) in enumerate(estrutura):
            par = self.d.add_paragraph()
            pf = par.paragraph_format; pf.space_after = Pt(0 if nivel == 2 else 1)
            pf.space_before = Pt(0 if nivel == 2 else 3); pf.line_spacing = 1.0
            pf.left_indent = Cm(0.0 if nivel == 1 else 0.8)
            pf.tab_stops.add_tab_stop(Cm(LARGURA_UTIL_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            if i == 0:
                for tipo in ("begin",):
                    rr = par.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tipo); rr._r.append(f)
                rr = par.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
                it.text = ' TOC \\o "1-2" \\u '; rr._r.append(it)
                rr = par.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "separate"); rr._r.append(f)
            pagina = str(self.paginas.get(texto, "0"))
            r = par.add_run(f"{texto}\t{pagina}")
            r.font.size = Pt(10.5 if nivel == 1 else 9.5); r.bold = (nivel == 1)
            if i == n - 1:
                rr = par.add_run(); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), "end"); rr._r.append(f)
        self.quebra()

    def salvar(self, caminho):
        self.d.save(caminho)
