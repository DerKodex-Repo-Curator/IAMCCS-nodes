const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const field = 'h3_fast_latent_stage2_native_windows';
class Element {
  constructor() { this.children = []; this.dataset = {}; this.style = {}; }
  append(...children) { this.children.push(...children); }
}
const node = { widgets: [{ name: field, value: false }, { name: 'steps', value: 6 }, { name: 'upscale_mode', value: 'h3_fast_latent_2pass' }] };
let refreshes = 0;
const context = {
  node, document: { createElement: () => new Element() },
  widget: (n, name) => n.widgets.find(w => w.name === name),
  choices: () => [], human: name => name,
  branchGate: () => ({ available: true }), omniLockedField: () => false,
  selectedAcceleration: () => 'native',
  setValue: (n, name, value) => { n.widgets.find(w => w.name === name).value = value; },
  refresh: () => { refreshes++; }, OPEN_H3_REFINE_FIELDS: [],
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('  function makeControl('), source.indexOf('  function warnings(')), context);
const box = context.makeControl(field), checkbox = box.children.find(c => c.type === 'checkbox');
assert.strictEqual(box.children[0].textContent, 'Native Stage-2 Windows · Experimental');
assert.strictEqual(checkbox.checked, false);
assert.match(box.children.at(-1).textContent, /CPU-safe accumulation/);
for (const value of [true, false]) {
  checkbox.checked = value;
  checkbox.onchange();
  assert.strictEqual(node.widgets[0].value, value);
}
assert.strictEqual(refreshes, 2);
assert.strictEqual(node.widgets[1].value, 6, 'Toggle must preserve sampling steps');
assert.strictEqual(node.widgets[2].value, 'h3_fast_latent_2pass');
vm.runInContext(source.slice(source.indexOf('const GROUPS ='), source.indexOf('const ENGINE_IDS =')) + '\nthis.groups = GROUPS;', context);
assert(context.groups.find(g => g.id === 'finish').fields.includes(field));
vm.runInContext(source.slice(source.indexOf('const FUNCTIONAL_LAYOUT ='), source.indexOf('const MODE_CHOICES =')) + '\nthis.layout = FUNCTIONAL_LAYOUT;', context);
assert(context.layout.finish.find(([name]) => name === 'FAST LATENT GPU BUDGET')[1].includes(field));
console.log('Native Stage-2 toggle changes its canonical widget without changing sampling or delivery; Output control PASS');
