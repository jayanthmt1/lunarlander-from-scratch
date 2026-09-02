#!/bin/sh
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
core=open('core.js').read()
core=core[:core.index("if (typeof module !== 'undefined') module.exports = {")].rstrip()+"\n"
core=core.replace("'use strict';\n","",1)
out=open('template.html').read().replace("/*__CORE__*/",core)
open('/tmp/_build.html','w').write(out)
i=out.index('<script>'); j=out.index('</script>')
open('/tmp/_build.js','w').write(out[i+8:j])
PY
node --check /tmp/_build.js
cp /tmp/_build.html lunar-reinforce.html
echo "built lunar-reinforce.html ($(wc -c < lunar-reinforce.html) bytes) — syntax OK"
