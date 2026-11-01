// Exercise the offline report's script with a minimal DOM/canvas test double.
// This validates data, event handlers and coordinates, not browser rendering.
const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'../..');
const report=path.join(root,'docs/offaxis-eda/corner-benchmark/report.html');
const page=fs.readFileSync(report,'utf8');
const source=page.match(/<script>([\s\S]*?)<\/script>/)[1];
let draws=0;
const context2d={drawImage(){draws++},beginPath(){},arc(){},stroke(){},moveTo(){},lineTo(){}};
const nodes={
 canvas:{getContext(){return context2d},style:{},getBoundingClientRect(){return{left:0,top:0,width:this.width,height:this.height}}},
 frame:{value:'0',appendChild(){}},method:{value:'quadrant_model'},kind:{value:'L',after(){}},
 zoom:{value:'1'},detail:{textContent:''}
};
class MockImage{set src(value){this.complete=true;this.onload()} }
const sandbox={document:{getElementById(id){return nodes[id]},createElement(tag){return{value:tag==='select'?'material':''}}},Image:MockImage,console};
vm.createContext(sandbox);new vm.Script(source).runInContext(sandbox);
const state=vm.runInContext(`(()=>{
 if(DATA.length<6||points.length!==630)throw Error('Missing frames or L points');
 let sample=points[0];canvas.onmousemove({clientX:sample.x,clientY:sample.y});
 if(!detail.textContent.includes('held-point residual'))throw Error('Hover not connected');
 zoom.value='4';zoom.oninput();if(canvas.width!==current.width*4)throw Error('Zoom scale');
 method.value='edge_lines';method.onchange();if(points.length!==630)throw Error('Edge selection');
 kind.value='X';kind.onchange();if(points.length!==0)throw Error('Unsupported X edge points');
 method.value='subpix';method.onchange();if(points.length!==125)throw Error('X selection');
 mode.value='grid';mode.onchange();frame.value='2';frame.onchange();
 if(!current.path.includes('.bmp'))throw Error('Frame selection');
 return {frames:DATA.length,embeddedPoints:DATA.reduce((a,x)=>a+x.points.length,0),frameSelection:true,
         methodSelection:true,kindSelection:true,projectionSelection:true,zoom:true,hover:true};
})()`,sandbox);
if(draws<7)throw Error('Canvas events were not exercised');
const result={...state,canvasRedraws:draws,browserRenderingVerified:false,
 note:'DOM/canvas test double; browser loopback preview was unavailable.'};
fs.writeFileSync(path.join(path.dirname(report),'report_validation.json'),JSON.stringify(result,null,2));
console.log(result);
