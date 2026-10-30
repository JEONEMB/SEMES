import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from 'file:///C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('../2026-10-29_중간.pptx'));
await fs.writeFile('inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,image,table',maxChars:1000000})).ndjson);
console.log(p.slides.items.length); console.log(p.slides.items[2].shapes.items.map(s=>({id:s.id,text:String(s.text),pos:s.position}))); console.log('TABLES',p.slides.items[3].tables.items.length);
await fs.writeFile('import.png',new Uint8Array(await (await p.export({slide:p.slides.items[1],format:'png',scale:1})).arrayBuffer()));

