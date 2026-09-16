import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(new URL("../src/web/index.html", import.meta.url), "utf8");
const script = html.split("<script>")[1].split("// ---------------- boot ----------------")[0];
const clone = value => JSON.parse(JSON.stringify(value));

function harness() {
  const requests = [], rendered = [], streams = [];
  const node = () => ({
    addEventListener() {}, appendChild() {}, remove() {}, focus() {},
    querySelector() { return null; },
    classList: { add() {}, remove() {}, toggle() {} }, style: {},
  });
  const context = vm.createContext({
    document: { getElementById: () => node(), createElement: node },
    localStorage: { setItem() {}, getItem() { return null; } },
    requestAnimationFrame: fn => fn(), rendered, streams,
    TextDecoder, encodeURIComponent,
    fetch: (url, args) => new Promise(resolve => requests.push({ url, args, resolve })),
    EventSource: class {
      constructor(url) { this.url = url; streams.push(this); }
      close() { this.closed = true; }
    },
  });
  vm.runInContext(script, context);
  vm.runInContext(`
    chats = [
      {id:'A', msgs:[], state:{dataset:'IFNG', hits:[], misses:[]}},
      {id:'B', msgs:[], state:{dataset:'IL2', hits:[], misses:[]}}
    ];
    current = chats[0]; appState = current.state;
    renderSidebar = () => {}; renderResult = r => rendered.push(r);
    addThinking = () => ({remove(){}}); refreshView = () => {};
    addYou = () => {}; addBot = () => {}; clearHero = () => {};
    makeRunHandler = (_tl, finish) => e => {if(e.type === 'done') finish();};
  `, context);
  return {
    context, requests, rendered, streams,
    run: code => vm.runInContext(code, context),
    async reply(index, result, ok = true) {
      requests[index].resolve({ ok, status: ok ? 200 : 400, json: async () => result });
      await new Promise(resolve => setImmediate(resolve));
    },
  };
}

test("responses update their original conversation after switching chats", async () => {
  const h = harness();
  const pending = h.run('ask("test")');
  h.run('selectChat("B")');
  await h.reply(0, { kind: "chat", text: "A reply", state: { dataset: "IFNG", hits: ["STAT1"], misses: [] } });
  await pending;
  assert.equal(h.run("current.id"), "B");
  assert.equal(h.run("appState.dataset"), "IL2");
  assert.equal(h.run("current.msgs.length"), 0);
  assert.deepEqual(clone(h.run('chats.find(c => c.id === "A").state.hits')), ["STAT1"]);
  assert.equal(h.rendered.length, 0);
  assert.ok(h.requests[0].args.headers["x-waddington-token"]);
});

test("late responses do not resurrect deleted chats or overwrite a new chat", async () => {
  const h = harness();
  const pending = h.run('ask("test")');
  h.run('deleteChat("A"); newChat()');
  await h.reply(0, { kind: "chat", text: "late", state: { dataset: "IFNG" } });
  await pending;
  assert.equal(h.run('chats.some(c => c.id === "A")'), false);
  assert.equal(h.run("current.msgs.length"), 0);
  assert.equal(h.run("appState.dataset"), "IL2");
});

test("finishing a background chat does not unlock another chat's in-flight request", async () => {
  const h = harness();
  const first = h.run('ask("A")');
  h.run('selectChat("B")');
  const second = h.run('ask("B")');
  await h.reply(0, { kind: "chat", text: "A" }); await first;
  assert.equal(h.run("send.disabled"), true);
  await h.reply(1, { kind: "chat", text: "B" }); await second;
  assert.equal(h.run("send.disabled"), false);
  assert.equal(h.run("current.msgs[0].result.text"), "B");
});

test("a background demo completes without unlocking or modifying the current chat", async () => {
  const h = harness();
  h.run('runDemo("IFNG", 1); selectChat("B")');
  const pending = h.run('ask("B")');
  assert.match(h.streams[0].url, /token=/);
  h.streams[0].onmessage({ data: JSON.stringify({ type: "done" }) });
  assert.equal(h.run("send.disabled"), true);
  assert.equal(h.run("appState.dataset"), "IL2");
  await h.reply(0, { kind: "chat", text: "B" }); await pending;
  assert.equal(h.run("send.disabled"), false);
});

test("HTTP errors stay in their originating chat without applying server state", async () => {
  const h = harness();
  const pending = h.run('ask("test")');
  h.run('selectChat("B")');
  await h.reply(0, { text: "Invalid results", state: { dataset: "BAD" } }, false); await pending;
  assert.equal(h.run('chats.find(c => c.id === "A").state.dataset'), "IFNG");
  assert.equal(h.run('chats.find(c => c.id === "A").msgs[0].result.text'), "Invalid results");
  assert.equal(h.run("current.msgs.length"), 0);
});
