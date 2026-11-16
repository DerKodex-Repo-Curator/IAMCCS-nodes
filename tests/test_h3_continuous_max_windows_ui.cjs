const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8').replace(/\r\n/g, '\n');
const start = source.indexOf('    if (active === "extension" && ["fl2va_extended_av", "t2va_continuous"]');
assert.ok(start > 0);
const controls = source.slice(start, source.indexOf('    if (active === "engine_multishot")', start));
const helpers = source.slice(source.indexOf('function restoreNamedValues('), source.indexOf('function normalizePianosequenzaStageControls('));
for (const mode of ['fl2va_extended_av', 't2va_continuous']) {
  const node = {widgets: [
    {name: 'extended_av_window_policy', value: 'max_native'},
    {name: 'extended_av_profile', value: 'safe_8_12gb'},
  ]};
  const buttons = [], elements = [];
  const widget = (n, name) => n.widgets.find(w => w.name === name);
  const context = vm.createContext({
    mode, active: 'extension', node, widget,
    document: {createElement: () => ({})},
    recipes: {append: el => elements.push(el)},
    addRecipe: (bar, label, id, enabled, selected, click) => buttons.push({label, id, enabled, selected, click}),
    setValue: (n, name, value) => {widget(n, name).value = value;},
    refresh: () => {},
  });
  vm.runInContext(controls, context);
  assert.equal(buttons.find(b => b.id === 'ext-window:max_native').selected, true);
  assert.ok(elements.some(e => /345-frame/.test(e.textContent)));
  buttons.find(b => b.id === 'ext-window:profile').click();
  assert.equal(widget(node, 'extended_av_window_policy').value, 'profile');
  buttons.length = 0; elements.length = 0;
  vm.runInContext(controls, context);
  assert.equal(buttons.find(b => b.id === 'ext-window:profile').selected, true);
  assert.ok(elements.some(e => /reduce memory/.test(e.textContent)));
  buttons.find(b => b.id === 'ext-window:max_native').click();
  const data = {input: {required: {
    extended_av_window_policy: [['max_native', 'profile'], {default: 'max_native'}],
    extended_av_profile: [['safe_8_12gb'], {default: 'safe_8_12gb'}],
  }}};
  Object.assign(context, {data, saved: {}});
  vm.runInContext(helpers + 'serializeNamedValues(node, saved, data);', context);
  widget(node, 'extended_av_window_policy').value = 'profile';
  vm.runInContext('restoreNamedValues(node, saved, data);', context);
  assert.equal(widget(node, 'extended_av_window_policy').value, 'max_native');
}
console.log('PASS: Continuous I2V/T2V max/profile buttons change the real setting and survive named serialization.');
