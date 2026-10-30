import fs from 'node:fs/promises';import {PresentationFile,FileBlob} from 'file:///C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('../2026-10-29_중간.pptx'));
for(const k of ['shapes','images','tables'])console.log(k,Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.items[3][k])));
console.log('shape',Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.items[3].shapes.items[0]))); console.log(p.slides.items.map((s,i)=>({i:i+1,sh:s.shapes.items.length,img:s.images.items.length,notes:String(s.speakerNotes.text).slice(0,35)})));
