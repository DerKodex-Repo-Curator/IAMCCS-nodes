const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const s = fs.readFileSync(require('node:path').join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8').replace(/\r\n/g, '\n');
const vbvr = s.slice(s.indexOf('function applyVbvrPreset('), s.indexOf('function human('));
const stop = vbvr.indexOf('\nfunction ', vbvr.indexOf('function choices(') + 10);
const code = stop < 0 ? vbvr : vbvr.slice(0, stop);
const node = {id: 811, widgets: Object.entries({
  secondary_lora_enabled: false, secondary_lora_name: 'VBVR_H3_attn_only.safetensors',
  secondary_lora_strength: 0.5, secondary_lora_continuation_only: true,
  acceleration: 'pika_h3_flex_6step', h3_video_decoder: 'native',
}).map(([name, value]) => ({name, value}))};
const widget = (n, name) => n.widgets.find(w => w.name === name);
widget(node, 'secondary_lora_name').options = {values: ['VBVR_H3_attn_only.safetensors', 'folder\\H3-Licon-VBVR-I2V.safetensors']};
const context = vm.createContext({node, widget, setValue: (n, key, value) => {widget(n, key).value = value;},
  document: {dispatchEvent() {}}, CustomEvent: class {}});
vm.runInContext(code + '\napplyVbvrPreset(node)', context);
assert.equal(widget(node, 'secondary_lora_name').value, 'folder\\H3-Licon-VBVR-I2V.safetensors');
assert.equal(widget(node, 'secondary_lora_enabled').value, true);
assert.equal(widget(node, 'secondary_lora_strength').value, 1);
assert.equal(widget(node, 'secondary_lora_continuation_only').value, false);
assert.equal(widget(node, 'acceleration').value, 'pika_h3_flex_6step');
widget(node, 'secondary_lora_name').options.values = ['VBVR_H3_attn_only.safetensors'];
assert.throws(() => vm.runInContext('applyVbvrPreset(node)', context), /complete/);
const helpers = s.slice(s.indexOf('function restoreNamedValues('), s.indexOf('function normalizePianosequenzaStageControls('));
Object.assign(context, {saved: {}, data: {input: {required: {h3_video_decoder: [['native', 'sol'], {default: 'native'}]}}}});
widget(node, 'h3_video_decoder').value = 'sol';
vm.runInContext(helpers + '\nserializeNamedValues(node, saved, data)', context);
widget(node, 'h3_video_decoder').value = 'native';
vm.runInContext('restoreNamedValues(node, saved, data)', context);
assert.equal(widget(node, 'h3_video_decoder').value, 'sol');
console.log('PASS: full VBVR preset uses the creative slot from segment 1, preserves acceleration, and Sol selection survives serialization.');
