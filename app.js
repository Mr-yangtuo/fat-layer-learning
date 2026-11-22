(function(){
'use strict';
const DATA=JSON.parse(document.getElementById('project-data').textContent);
const files=DATA.files;
const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let currentFile=null,currentSegment=null,mode='report';
function toast(msg){$('#toast').textContent=msg;$('#toast').style.opacity='1';clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').style.opacity='0',2300)}
function highlight(line){const re=/(#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|class|return|import|from|as|if|elif|else|for|in|while|with|try|except|raise|and|or|not|is|lambda|pass|None|True|False|super|yield|finally|break|continue|assert|global|exec|then|fi|do|done|function|local|export)\b|\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)/g;let out='',pos=0;for(const m of line.matchAll(re)){out+=esc(line.slice(pos,m.index));const t=m[0],cl=t[0]==='#'?'comment':/^["']/.test(t)?'str':/^\d/.test(t)?'num':'kw';out+='<span class="'+cl+'">'+esc(t)+'</span>';pos=m.index+t.length}return out+esc(line.slice(pos))}
function codeBlock(f,start,end){return '<pre><code>'+f.lines.slice(start-1,end).map((l,i)=>'<span class="code-line" data-code-line="'+(start+i)+'"><span class="ln">'+(start+i)+'</span><span class="code-text">'+highlight(l)+'</span></span>').join('')+'</code></pre>'}
let figureWidth=1100,figureScale=1;
function setFigureScale(value){figureScale=Math.max(.2,Math.min(2.5,value));$('#figure-image').style.width=(figureWidth*figureScale)+'px';$('#figure-scale').textContent=Math.round(figureScale*100)+'%'}
$$('[data-zoom-figure]').forEach(button=>button.onclick=()=>{const fig=$('#figure-'+button.dataset.zoomFigure),img=fig.querySelector('img');$('#figure-title').textContent=fig.querySelector('.figure-top span').textContent;$('#figure-image').src=img.src;$('#figure-image').alt=img.alt;figureWidth=Number(img.getAttribute('width'))||1100;$('#figure-dialog').showModal();setFigureScale(Math.min(1,Math.max(.75,($('#figure-stage').clientWidth-32)/figureWidth)));$('#figure-stage').scrollTop=0;$('#figure-stage').scrollLeft=0;});
$('#figure-close').onclick=()=>$('#figure-dialog').close();
$('#figure-fit').onclick=()=>setFigureScale(($('#figure-stage').clientWidth-32)/figureWidth);
$('#figure-minus').onclick=()=>setFigureScale(figureScale-.15);
$('#figure-plus').onclick=()=>setFigureScale(figureScale+.15);
const sections=$$('.section');
const reportToc='<a href="#report/intro"><b>00</b>项目概览</a>'+sections.map((s,i)=>'<a href="#report/'+s.id+'"><b>'+String(i+1).padStart(2,'0')+'</b>'+s.dataset.title+'</a>').join('');
function treeHTML(){const root={};files.forEach(f=>{const parts=f.path.split('/');let node=root;parts.forEach((p,i)=>{if(i===parts.length-1)node[p]=f;else node=node[p]||(node[p]={})})});function walk(obj){return Object.entries(obj).sort((a,b)=>{const ad=!a[1].path,bd=!b[1].path;return ad===bd?a[0].localeCompare(b[0]):ad?-1:1}).map(([name,value])=>value.path?'<a class="tree-file" href="#code/'+encodeURI(value.path)+'" data-file="'+esc(value.path)+'"><span>'+(/\.py$/.test(name)?'PY':'SH')+'</span>'+esc(name)+'</a>':'<details class="tree-folder" open><summary>▱ '+esc(name)+'</summary>'+walk(value)+'</details>').join('')}return '<a id="project-guide-link" class="project-guide-link" href="#code/guide">整体导读与阅读顺序</a><p class="tree-intro">主程序串起数据、模型与损失。先读导读中的主线，再按目录补齐细节。</p><div class="tree-map-label">文件地图</div>'+walk(root)+'<p class="coverage-note">完整核心 .py / .sh 文件。运行产物、权重、缓存和原始样本不属于源码，未嵌入。</p>'}
function updateCounts(){$('#source-summary').textContent=files.length+' 个文件 · '+files.reduce((n,f)=>n+f.segments.length,0)+' 段讲解';$('#hero-file-count').textContent=files.length}
function switchMode(next){mode=next;document.body.dataset.mode=mode;if(mode==='report')document.body.classList.remove('reader-wide');$('#report-view').hidden=mode!=='report';$('#code-view').hidden=mode!=='code';$$('.mode-tabs button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));$('.side-label').innerHTML=mode==='report'?'阅读目录 <span>CONTENTS</span>':'核心源码 <span>FILE EXPLORER</span>';$('#toc').innerHTML=mode==='report'?reportToc:treeHTML();$('#sidebar').classList.remove('open')}
/*__READER__*/
function navigate(){
  const hash=decodeURI(location.hash.slice(1)||'report/intro');
  if(hash.startsWith('code/')){
    switchMode('code');const [path,query='']=hash.slice(5).split('?');
    if(path==='guide'||!path){showProjectGuide();return}
    renderFile(path);if(!currentFile)return;
    const line=Number(new URLSearchParams(query).get('line'));
    const found=line?currentFile.segments.findIndex(s=>s.start<=line&&s.end>=line):0;
    openSegment(found>=0?found:0,{updateURL:false,reveal:false});
    if(line)requestAnimationFrame(revealWorkbench);else window.scrollTo(0,0);
  }else{
    switchMode('report');const id=hash.replace(/^report\//,'');
    requestAnimationFrame(()=>document.getElementById(id)?.scrollIntoView({behavior:'instant',block:'start'}));
  }
}
function go(hash){if(location.hash===hash)navigate();else location.hash=hash}
$$('[data-mode]').forEach(b=>b.onclick=()=>go(b.dataset.mode==='report'?'#report/intro':'#code/guide'));
window.addEventListener('hashchange',()=>{$$('dialog[open]').forEach(dialog=>dialog.close());navigate()});
$('#menu-toggle').onclick=()=>$('#sidebar').classList.toggle('open');
function search(){const q=$('#search-input').value.trim().toLowerCase();if(!q){$('#search-results').innerHTML='<div class="empty-search">试试 soft_argmax、valid、数据泄漏、AdamW 或任意源码参数。<br>同时搜索报告、原始代码与逐段解释。</div>';return}const results=[];if($('#reader-guide').textContent.toLowerCase().includes(q))results.push({title:'全局导读与阅读顺序',sub:'源码研读 · 文件地图',hash:'#code/guide'});sections.forEach(s=>{if((s.textContent+s.dataset.title).toLowerCase().includes(q))results.push({title:s.dataset.title,sub:'项目汇报 · '+s.id,hash:'#report/'+s.id})});files.forEach(f=>{if((f.path+' '+f.overview+' '+JSON.stringify(f.components)+' '+JSON.stringify(f.readingGuide)).toLowerCase().includes(q))results.push({title:f.title,sub:f.path,hash:'#code/'+f.path});f.segments.forEach(s=>{if((JSON.stringify(s)+' '+f.lines.slice(s.start-1,s.end).join('\n')).toLowerCase().includes(q))results.push({title:s.title,sub:f.path+' · L'+s.start+'–'+s.end,hash:'#code/'+f.path+'?line='+s.start})})});$('#search-results').innerHTML=results.length?results.slice(0,70).map((r,i)=>'<button class="search-result" data-result="'+i+'">'+esc(r.title)+'<small>'+esc(r.sub)+'</small></button>').join(''):'<div class="empty-search">没有匹配结果，试试函数名或更短的关键词。</div>';$$('[data-result]').forEach(b=>b.onclick=()=>{$('#search-dialog').close();go(results[Number(b.dataset.result)].hash)})}
function openSearch(){$('#search-dialog').showModal();search();$('#search-input').focus()}
$('#search-open').onclick=openSearch;$('#search-close').onclick=()=>$('#search-dialog').close();$('#search-input').oninput=search;
document.addEventListener('keydown',e=>{if(e.key==='Escape'){const opened=$$('dialog[open]').at(-1);if(opened){e.preventDefault();opened.close()}return}if(e.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();openSearch()}});
function download(text,name,type){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function copy(text){try{await navigator.clipboard.writeText(text);toast('已复制完整源码')}catch{const t=document.createElement('textarea');t.value=text;document.body.append(t);t.select();const ok=document.execCommand('copy');t.remove();toast(ok?'已复制完整源码':'当前浏览器限制复制，请使用下载文件')}}
function softDemo(){const t=Number($('#temp').value),peak=Number($('#peak').value);$('#temp-value').textContent=t.toFixed(2);$('#peak-value').textContent=peak.toFixed(1);const z=Array.from({length:32},(_,y)=>5*Math.exp(-((y-9)**2)/8)+peak*Math.exp(-((y-23)**2)/8)),max=Math.max(...z),ex=z.map(a=>Math.exp((a-max)/t)),sum=ex.reduce((a,b)=>a+b,0),p=ex.map(a=>a/sum),expected=p.reduce((a,b,i)=>a+b*i,0),arg=z.indexOf(max),scale=150/Math.max(...p);let html='<path d="M20 175H410" stroke="#52716d"/>';p.forEach((a,i)=>html+='<rect x="'+(22+i*12)+'" y="'+(175-a*scale)+'" width="9" height="'+(a*scale)+'" rx="2" fill="#81ba9f"/>');const x=26+expected*12;html+='<path d="M'+x+' 15V180" stroke="#d6e879" stroke-width="2" stroke-dasharray="4 4"/><g fill="#a4beb4" font-size="11"><text x="20" y="197">0</text><text x="355" y="197">高度 y → 31</text></g>';$('#soft-chart').innerHTML=html;$('#soft-output').textContent='加权平均 '+expected.toFixed(2)+' 像素 / 最大峰 '+arg+' 像素'}

function interpolationDemo(){
const n=Number($('#interp-column').value),x=n/2,valid=x>=1&&x<=6;
let y,detail;
if(x<1){y=6;detail='首点之外 → 暂填端点高度 6';}
else if(x>6){y=8;detail='末点之外 → 暂填端点高度 8';}
else if(x<=3){const t=(x-1)/2;y=6+t*4;detail='t = ('+x+' − 1) / 2 = '+t.toFixed(3)+'<br>y = 6 + t × 4 = '+y.toFixed(3);}
else{const t=(x-3)/3;y=10-t*2;detail='t = ('+x+' − 3) / 3 = '+t.toFixed(3)+'<br>y = 10 − t × 2 = '+y.toFixed(3);}
const label=y/2;
$('#interp-column-value').textContent=n;
$('#interp-calculation').innerHTML='新列 '+n+' → 原位置 '+x.toFixed(1)+'<br>'+detail+'<br>新高度 = '+y.toFixed(3)+' × 0.5 = <strong>'+label.toFixed(3)+'</strong>';
$('#interp-calculation').dataset.height=String(label);
$('#interp-validity').textContent=valid?'有效：查询位置在 [1,6] 内，参与坐标监督。':'无效：填充值保留在数组里，但不参与坐标监督。';
$('#interp-validity').dataset.valid=String(valid);
const px=v=>35+v*48,py=v=>35+v*14,col=valid?'#d6e879':'#f0ab73';
let svg='<g stroke="#426461" stroke-width=".5">';
for(let k=0;k<=7;k++)svg+='<path d="M'+px(k)+' 35V210"/>';
for(let k=0;k<=12;k+=2)svg+='<path d="M35 '+py(k)+'H405"/>';
svg+='</g><path d="M35 25V210H410" fill="none" stroke="#a7c3b4"/><path d="M'+px(1)+' '+py(6)+'L'+px(3)+' '+py(10)+'L'+px(6)+' '+py(8)+'" fill="none" stroke="#8dd6b6" stroke-width="3"/><path d="M35 '+py(6)+'H'+px(1)+'M'+px(6)+' '+py(8)+'H405" stroke="#d09569" stroke-dasharray="5 4"/>';
[[1,6],[3,10],[6,8]].forEach(a=>{svg+='<circle cx="'+px(a[0])+'" cy="'+py(a[1])+'" r="4" fill="#8dd6b6"/><text x="'+(px(a[0])+8)+'" y="'+(py(a[1])-7)+'" fill="#d5e8df" font-size="11">('+a.join(',')+')</text>'});
svg+='<path d="M'+px(x)+' 35V'+py(y)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="4 4"/><circle cx="'+px(x)+'" cy="'+py(y)+'" r="6" fill="'+col+'" stroke="#142f32" stroke-width="2"/><g fill="#bfd3c9" font-size="11"><text x="8" y="20">y↓</text><text x="35" y="230">0</text><text x="347" y="230">原图 x →</text><text x="'+Math.min(340,Math.max(38,px(x)-20))+'" y="25">查询 x='+x+'</text></g>';
$('#interp-chart').innerHTML=svg;
}
$('#interp-column').oninput=interpolationDemo;interpolationDemo();

$('#temp').oninput=softDemo;$('#peak').oninput=softDemo;softDemo();updateCounts();navigate();
let scrollPending=false;window.addEventListener('scroll',()=>{if(mode!=='report'||scrollPending)return;scrollPending=true;requestAnimationFrame(()=>{let active='intro';sections.forEach(s=>{if(s.getBoundingClientRect().top<180)active=s.id});$$('#toc a').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#report/'+active));scrollPending=false})},{passive:true});
})();
