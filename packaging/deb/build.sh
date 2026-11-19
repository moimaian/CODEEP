#!/bin/bash
# Monta o pacote codeep_<versão>_all.deb a partir do estado atual do repo.
#
# Uso:
#   packaging/deb/build.sh [versão]
#
# A versão segue MAJOR.MINOR.PATCH-N (ver packaging/VERSIONING.md antes de
# escolher o próximo número: PATCH para bug fix, MINOR para funcionalidade
# nova sem quebrar nada, MAJOR para mudança incompatível; N é a revisão de
# empacotamento, só sobe sozinha se for uma mudança que não toca em
# CODEEP.py, CODEEP_requirements.py nem MODULES/). Sem argumento, usa
# versao.py (fonte única — é o mesmo valor mostrado no rodapé da aba INÍCIO
# e no atalho .desktop). Lembre de:
# 1) atualizar versao.py, 2) adicionar a entrada correspondente no topo de
# packaging/deb/changelog, antes de rodar este script.
#
# O .deb resultante é gravado em packaging/dist/ (ignorada pelo git).

set -euo pipefail
umask 022   # garante dirs 755 / arquivos 644 no staging, independente do umask local

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/packaging/dist"
VERSAO_CODIGO="$(sed -n 's/^VERSAO = "\(.*\)"$/\1/p' "$PROJECT_ROOT/versao.py")"
VERSAO_CHANGELOG="$(sed -n '1s/^codeep (\(.*\)).*/\1/p' "$SCRIPT_DIR/changelog")"
if [ -z "$VERSAO_CODIGO" ] || [ "$VERSAO_CODIGO" != "$VERSAO_CHANGELOG" ]; then
    echo "build.sh: versao.py ($VERSAO_CODIGO) e o topo de packaging/deb/changelog" >&2
    echo "          ($VERSAO_CHANGELOG) estão em versões diferentes — atualize os dois" >&2
    echo "          antes de empacotar (ver packaging/VERSIONING.md)." >&2
    exit 1
fi
VERSION="${1:-$VERSAO_CODIGO}"

for tool in dpkg-deb rsync gzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "build.sh: dependência ausente: $tool" >&2
        exit 1
    fi
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Preparando staging em $STAGE"
mkdir -p \
    "$STAGE/DEBIAN" \
    "$STAGE/opt/codeep" \
    "$STAGE/usr/bin" \
    "$STAGE/usr/share/applications" \
    "$STAGE/usr/share/doc/codeep" \
    "$STAGE/usr/share/lintian/overrides"

# Só o CÓDIGO e os materiais de apoio vão no pacote. Ficam de fora, de propósito:
#   data/, checkpoints/, hpo/, PROJECTS/ — dados e modelos do usuário (gigabytes; e o
#       launcher sincroniza /opt/codeep sobre ~/CODEEP, o que sobrescreveria um modelo
#       treinado pelo usuário);
#   tools/ — xtb e P2Rank, que a própria janela de dependências baixa para ~/CODEEP/tools;
#   TEST/, results/, ARTIGO/ — benchmarks e material de desenvolvimento.
echo "==> Copiando arquivos da aplicação para /opt/codeep"
ITENS=(
    "$PROJECT_ROOT/CODEEP.py"
    "$PROJECT_ROOT/CODEEP_requirements.py"
    "$PROJECT_ROOT/versao.py"
    "$PROJECT_ROOT/MODULES"
    "$PROJECT_ROOT/ICONS"
    "$PROJECT_ROOT/TUTORIALS"
)
# LICENSE.txt e README.md entram quando existirem no repositório.
for opcional in LICENSE.txt LICENSE README.md; do
    [ -f "$PROJECT_ROOT/$opcional" ] && ITENS+=("$PROJECT_ROOT/$opcional")
done
rsync -a \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='_build/' \
    "${ITENS[@]}" \
    "$STAGE/opt/codeep/"

# Normaliza permissões: o diretório de trabalho local pode ter modos
# incomuns; o pacote precisa ser legível/executável por qualquer usuário do sistema.
find "$STAGE/opt/codeep" -type d -exec chmod 755 {} +
find "$STAGE/opt/codeep" -type f -exec chmod 644 {} +
chmod 755 "$STAGE/opt/codeep/CODEEP.py"

echo "==> Instalando launcher e menu"
install -m 755 "$SCRIPT_DIR/codeep-launcher" "$STAGE/usr/bin/codeep"
install -m 644 "$SCRIPT_DIR/CODEEP.desktop" "$STAGE/usr/share/applications/CODEEP.desktop"
# copyright no formato DEP-5, apontando para /usr/share/common-licenses/GPL-3:
install -m 644 "$SCRIPT_DIR/copyright" "$STAGE/usr/share/doc/codeep/copyright"
install -m 644 "$SCRIPT_DIR/lintian-overrides" "$STAGE/usr/share/lintian/overrides/codeep"
[ -f "$PROJECT_ROOT/README.md" ] && \
    install -m 644 "$PROJECT_ROOT/README.md" "$STAGE/usr/share/doc/codeep/README.md"
gzip -9nc "$SCRIPT_DIR/changelog" > "$STAGE/usr/share/doc/codeep/changelog.Debian.gz"
chmod 644 "$STAGE/usr/share/doc/codeep/changelog.Debian.gz"

echo "==> Gerando DEBIAN/control (versão $VERSION)"
INSTALLED_SIZE="$(du -sk "$STAGE/opt/codeep" | cut -f1)"
sed \
    -e "s/@VERSION@/$VERSION/" \
    -e "s/@INSTALLED_SIZE@/$INSTALLED_SIZE/" \
    "$SCRIPT_DIR/control.in" > "$STAGE/DEBIAN/control"

install -m 755 "$SCRIPT_DIR/postinst" "$STAGE/DEBIAN/postinst"
install -m 755 "$SCRIPT_DIR/postrm" "$STAGE/DEBIAN/postrm"

mkdir -p "$DIST_DIR"
OUT_DEB="$DIST_DIR/codeep_${VERSION}_all.deb"

echo "==> Empacotando $OUT_DEB"
dpkg-deb --root-owner-group --build "$STAGE" "$OUT_DEB"

if command -v lintian >/dev/null 2>&1; then
    echo "==> lintian (informativo, não interrompe o build)"
    lintian "$OUT_DEB" || true
fi

echo "==> Pronto: $OUT_DEB"
