/* Source reader. Embedded inside app.js's closure by build.py. */
let graphSerial=0;
function coreBadge(segment){return segment.core?'<span class="core-badge" title="处理与训练主线的核心代码">★ 核心</span>':''}
function paragraphs(value){return (Array.isArray(value)?value:[value]).filter(Boolean).map(p=>'<p>'+esc(p)+'</p>').join('')}
function readerList(items){return '<ul>'+items.map(item=>'<li>'+esc(item)+'</li>').join('')+'</ul>'}
function wrapGraphText(text,width){
  const lines=[];let line='',used=0;
  for(const character of String(text||'')){const size=character.charCodeAt(0)>255?1:.57;if(character==='\n'||used+size>width){lines.push(line);line='';used=0;if(character==='\n')continue}line+=character;used+=size;}
  if(line)lines.push(line);return lines;
}
function graphSVG(graph){
  const id='reader-flow-'+(++graphSerial),nodes=graph.nodes,lookup=new Map(nodes.map((n,i)=>[n.id,i]));
  const ranks=nodes.map(()=>0),states=nodes.map(()=>0),order=[],feedback=new Set();
  function visit(i){states[i]=1;graph.edges.filter(e=>lookup.get(e.from)===i).forEach(edge=>{const next=lookup.get(edge.to);if(states[next]===1)feedback.add(edge);else if(!states[next])visit(next)});states[i]=2;order.push(i)}
  nodes.forEach((n,i)=>{if(!states[i])visit(i)});
  order.reverse().forEach(i=>graph.edges.forEach(edge=>{const a=lookup.get(edge.from),b=lookup.get(edge.to);if(a===i&&!feedback.has(edge))ranks[b]=Math.max(ranks[b],ranks[a]+1)}));
  const levels=[];nodes.forEach((node,i)=>(levels[ranks[i]]||(levels[ranks[i]]=[])).push(i));
  const detours=graph.edges.filter(e=>feedback.has(e)||ranks[lookup.get(e.to)]>ranks[lookup.get(e.from)]+1);
  const nodeW=242,gapX=42,margin=32,innerW=Math.max(700,Math.max(...levels.map(l=>l.length))*(nodeW+gapX)-gapX);
  const sideSpace=detours.length?70:0,origin=margin+sideSpace,width=innerW+margin*2+sideSpace*2,positions=[];
  let y=32;
  levels.forEach(level=>{
    const layout=level.map(i=>({i,title:wrapGraphText(nodes[i].title,15),detail:wrapGraphText(nodes[i].detail,16)}));
    const rowH=Math.max(...layout.map(n=>32+n.title.length*22+n.detail.length*21));
    layout.forEach((n,j)=>positions[n.i]={x:origin+(innerW-(level.length*nodeW+(level.length-1)*gapX))/2+j*(nodeW+gapX),y,w:nodeW,h:rowH,...n});
    y+=rowH+78;
  });
  const height=y-46;
  let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+width+' '+height+'" width="'+width+'" height="'+height+'" role="img" aria-labelledby="'+id+'-title '+id+'-desc"><title id="'+id+'-title">'+esc(graph.title)+'</title><desc id="'+id+'-desc">'+esc(nodes.map(n=>n.title+'：'+n.detail).join('；'))+'</desc><defs><marker id="'+id+'-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#4b786d"/></marker></defs><rect width="100%" height="100%" fill="#fcfdfb"/>';
  const edgeLabels=[];
  graph.edges.forEach((edge,ei)=>{
    const a=positions[lookup.get(edge.from)],b=positions[lookup.get(edge.to)];let d,lx,ly;
    if(detours.includes(edge)){
      const onLeft=a.x+a.w/2<origin+innerW/2,offset=24+(detours.indexOf(edge)%3)*16;
      const rail=onLeft?origin-offset:origin+innerW+offset,sx=a.x+a.w*(onLeft?.15:.85),sy=a.y+a.h,tx=b.x+b.w/2,ty=b.y;
      const exitY=sy+24+(detours.indexOf(edge)%2)*12,enterY=ty-24;
      d='M'+sx+' '+sy+'V'+exitY+'H'+rail+'V'+enterY+'H'+tx+'V'+(ty-2);
      lx=onLeft?rail+6:sx+15;ly=exitY-8;
    }else{
      const sx=a.x+a.w/2,sy=a.y+a.h,tx=b.x+b.w/2,ty=b.y,mid=sy+(ty-sy)/2;
      d='M'+sx+' '+sy+'V'+mid+'H'+tx+'V'+(ty-2);
      lx=sx===tx?sx+14:(sx+tx)/2;ly=mid-9;
    }
    svg+='<path d="'+d+'" fill="none" stroke="#4b786d" stroke-width="1.7" marker-end="url(#'+id+'-arrow)"/>';
    if(edge.label)edgeLabels.push({x:lx,y:ly,text:edge.label});
  });
  positions.forEach(n=>{
    const kind=nodes[n.i].kind||'process',fill=kind==='decision'?'#f2eefa':kind==='input'?'#e5f1ef':kind==='output'?'#eaf2da':'#f1f5f2';
    const stroke=kind==='decision'?'#ada0c9':'#b6cdc4';
    const outline=kind==='decision'?'<path d="M'+(n.x+18)+' '+n.y+'H'+(n.x+n.w-18)+'L'+(n.x+n.w)+' '+(n.y+n.h/2)+'L'+(n.x+n.w-18)+' '+(n.y+n.h)+'H'+(n.x+18)+'L'+n.x+' '+(n.y+n.h/2)+'Z"':'<rect x="'+n.x+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'" rx="10"';
    svg+=outline+' fill="'+fill+'" stroke="'+stroke+'"/>';
    svg+='<g text-anchor="middle" font-family="Segoe UI,Microsoft YaHei,sans-serif" fill="#193e38">';
    n.title.forEach((line,j)=>svg+='<text x="'+(n.x+n.w/2)+'" y="'+(n.y+25+j*22)+'" font-size="15" font-weight="600">'+esc(line)+'</text>');
    n.detail.forEach((line,j)=>svg+='<text x="'+(n.x+n.w/2)+'" y="'+(n.y+29+n.title.length*22+j*21)+'" font-size="14" fill="#526c65">'+esc(line)+'</text>');
    svg+='</g>';
  });
  edgeLabels.forEach(label=>{const lines=wrapGraphText(label.text,18),w=Math.max(...lines.map(l=>Array.from(l).reduce((n,c)=>n+(c.charCodeAt(0)>255?14:7.5),0)))+12;svg+='<g font-family="Microsoft YaHei,sans-serif" font-size="13" fill="#365d52"><rect x="'+(label.x-6)+'" y="'+(label.y-15)+'" width="'+w+'" height="'+(lines.length*18+4)+'" rx="4" fill="#fcfdfb"/>';lines.forEach((line,j)=>svg+='<text x="'+label.x+'" y="'+(label.y+j*18)+'">'+esc(line)+'</text>');svg+='</g>'});
  return svg+'</svg>';
}
function flowFigure(graph){
  if(!graph||!graph.nodes?.length)return '';
  return '<figure class="reader-flow"><div class="reader-flow-head"><strong>'+esc(graph.title)+'</strong><button type="button" data-flow-zoom>放大流程图</button></div><div class="reader-flow-canvas" tabindex="0" role="region" aria-label="'+esc(graph.title)+'，可放大查看">'+graphSVG(graph)+'</div><figcaption>'+esc(graph.caption||'按箭头阅读数据与控制的流向。')+'<span>流程图随栏宽缩放，可放大阅读。</span></figcaption></figure>';
}
function parameterTable(rows){
  if(!rows?.length)return '<p class="reader-muted">本片段没有新增函数入参；使用到的值见下方变量说明。</p>';
  return '<div class="reader-table-wrap"><table class="reader-table parameter-table"><thead><tr><th scope="col">参数 / 类型</th><th scope="col">默认值</th><th scope="col">含义、用途与影响</th></tr></thead><tbody>'+rows.map(p=>'<tr><th scope="row"><code>'+esc(p.name)+'</code><small>'+esc(p.type)+'</small></th><td><code>'+esc(p.default??'必填')+'</code></td><td>'+esc(p.meaning)+'</td></tr>').join('')+'</tbody></table></div>';
}
function variableTable(rows){
  if(!rows?.length)return '<p class="reader-muted">本段没有独立的局部数据处理，相关名称在语法说明中解释。</p>';
  return '<div class="reader-table-wrap"><table class="reader-table variable-table"><thead><tr><th scope="col">变量 / 数据结构</th><th scope="col">从哪里来，用来做什么</th></tr></thead><tbody>'+rows.map(v=>'<tr><th scope="row"><code>'+esc(v.name)+'</code><small>'+esc(v.type)+'</small></th><td>'+esc(v.meaning)+'</td></tr>').join('')+'</tbody></table></div>';
}
function componentTable(f){
  return '<div class="reader-table-wrap"><table class="reader-table component-table"><thead><tr><th scope="col">函数 / 类 / 代码部分</th><th scope="col">分工与联系</th><th scope="col">对应讲解</th></tr></thead><tbody>'+f.components.map(c=>'<tr><th scope="row"><code>'+esc(c.name)+'</code><small>'+esc(c.kind)+'</small></th><td>'+esc(c.summary)+'</td><td><button data-open-line="'+c.segmentStart+'">L'+c.segmentStart+' 讲解</button></td></tr>').join('')+'</tbody></table></div>';
}
function typeGuide(f){
  if(!/torch|numpy|Tensor/.test(f.source))return '';
  return '<details class="definition-index type-guide"><summary>类型与张量尺寸怎么读</summary><div><p>Tensor 是 PyTorch 的多维数组，ndarray 是 NumPy 的多维数组。float 表示可带小数的数值，int 表示整数，bool 只有 True / False；list 是列表，tuple 是元组，dict 是按键名取值的字典。</p><p>形状中的 B 表示一批图像的数量，C 表示特征通道数，L 表示边界条数，H 表示高度，W 表示宽度。例如 [B,L,H,W] 是每张图、每条边界各有一张高度为 H、宽度为 W 的评分图；[B,L,W] 则是每条边界每列各有一个纵坐标。</p><p>项目的图像尺寸参数通常写成 (W,H)，图像张量的空间维度通常写成 [H,W]。每个函数的具体约定以参数表为准。</p></div></details>';
}
function visualFigure(id){
  const visual=DATA.visuals?.find(v=>v.id===id);if(!visual)return '';
  return '<figure class="concept-figure"><div class="concept-heading"><strong>'+esc(visual.title)+'</strong><button type="button" data-visual-zoom="'+esc(id)+'" aria-label="放大：'+esc(visual.title)+'">放大图解</button></div><button class="concept-image" data-visual-zoom="'+esc(id)+'" aria-label="查看大图：'+esc(visual.title)+'"><img src="'+visual.uri+'" alt="'+esc(visual.alt)+'" width="'+visual.width+'" height="'+visual.height+'" loading="lazy"></button><figcaption>'+esc(visual.caption)+'</figcaption></figure>';
}
function showProjectGuide(){
  currentFile=null;currentSegment=null;$('#reader-guide').hidden=false;$('#file-view').hidden=true;
  document.body.classList.remove('reader-wide');
  $('#project-guide-link')?.classList.add('active');window.scrollTo(0,0);
}
function revealWorkbench(){
  const workbench=$('#reader-workbench');if(!workbench)return;
  const headerHeight=$('.header').getBoundingClientRect().height;
  window.scrollTo({top:window.scrollY+workbench.getBoundingClientRect().top-headerHeight-10,behavior:'instant'});
}
function renderFile(path){
  const f=files.find(f=>f.path===path);currentFile=f||null;currentSegment=null;
  $('#reader-guide').hidden=true;$('#file-view').hidden=false;
  if(!f){$('#file-view').innerHTML='<h2>未找到这个文件</h2><p><a href="'+codeHref('guide')+'">返回全局导读与文件地图</a></p>';return}
  const coreCount=f.segments.filter(s=>s.core).length;
  let html='<nav class="reader-breadcrumb" aria-label="当前位置"><a href="'+codeHref('guide')+'">全局导读与阅读顺序</a><span>/</span><code>'+esc(f.path)+'</code></nav><article class="file-guide file-guide-compact"><div class="file-heading"><div><span class="reader-kicker">文件导读</span><h1>'+esc(f.title)+'</h1></div><div class="source-actions"><button id="copy-file">复制源码</button><button id="download-file">下载文件</button></div></div><div class="file-purpose"><div class="file-role">'+esc(f.role)+'</div>'+paragraphs(f.overview)+'</div><details class="file-details"><summary>查看这个文件的输入输出、函数分工和依赖关系</summary><div class="file-contract"><section><h3>接收什么</h3>'+readerList(f.inputs)+'</section><section><h3>产生什么</h3>'+readerList(f.outputs)+'</section></div><section class="file-components"><h3>主要函数与类怎样分工</h3>'+componentTable(f)+'</section><div class="file-reading"><h3>放回项目中理解</h3>'+readerList(f.readingGuide)+'</div>'+typeGuide(f);
  if(f.pipeline)html+=flowFigure(f.pipeline);
  if(f.dependencies?.length)html+='<div class="file-dependencies"><span>直接依赖的项目文件</span>'+f.dependencies.map(p=>'<a href="'+codeHref(p)+'">'+esc(p)+'</a>').join('')+'</div>';
  if(f.definitions?.length)html+='<details class="definition-index"><summary>全部函数与类的定义 · '+f.definitions.length+' 项</summary><div>'+f.definitions.map(d=>'<button data-open-line="'+d.segmentStart+'"><code>'+esc(d.signature)+'</code><span>L'+d.line+' · '+esc(d.scope||'文件级')+'</span></button>').join('')+'</div></details>';
  html+='</details></article><section class="reader-workbench" id="reader-workbench" aria-label="源码与解析对照阅读"><div class="workbench-toolbar"><div><strong>源码与解析对照</strong><span>'+f.lines.length+' 行 · '+f.segments.length+' 个代码段 · '+coreCount+' 段核心</span></div><button id="reader-focus">'+(document.body.classList.contains('reader-wide')?'显示文件目录':'展开阅读区')+'</button></div><p class="workbench-legend"><strong>★ 核心</strong> 数据处理、模型、损失、训练与验证；工程支撑与备用实现不标星。</p><div class="workbench-columns"><section class="source-pane" aria-label="左侧源码"><header class="pane-header"><div><h2>源码</h2><span>点击代码段，右侧同步解释</span></div><label class="segment-selector"><span class="sr-only">选择代码段</span><select id="segment-picker">'+f.segments.map((s,i)=>'<option value="'+i+'">'+(s.core?'★ 核心 · ':'')+'L'+s.start+'–'+s.end+' · '+esc(s.title)+'</option>').join('')+'</select></label></header><div class="source-scroll" id="source-scroll" tabindex="0" aria-label="完整源码，可独立滚动">';
  let cursor=1;
  f.segments.forEach((s,i)=>{
    if(s.start>cursor)html+='<div class="source-context"><div class="context-label">上下文 · L'+cursor+'–'+(s.start-1)+'</div>'+codeBlock(f,cursor,s.start-1)+'</div>';
    html+='<article class="source-segment'+(s.core?' is-core':'')+'" id="seg-'+s.start+'" data-segment="'+i+'"><button class="segment-select" data-select-segment="'+i+'" aria-pressed="false"><span>L'+s.start+'–'+s.end+'</span><strong>'+coreBadge(s)+esc(s.title)+'</strong></button>'+codeBlock(f,s.start,s.end)+'</article>';
    cursor=s.end+1;
  });
  if(cursor<=f.lines.length)html+='<div class="source-context"><div class="context-label">剩余上下文</div>'+codeBlock(f,cursor,f.lines.length)+'</div>';
  html+='</div></section><section class="analysis-pane" aria-label="右侧解析"><header class="pane-header analysis-header"><div><h2>图解与解析</h2><span id="selection-status" role="status" aria-live="polite"></span></div><div class="analysis-navigation"><button id="previous-segment" aria-label="选择上一代码段">上一段</button><button id="next-segment" aria-label="选择下一代码段">下一段</button></div></header><div class="analysis-scroll" id="analysis-content" tabindex="0" aria-label="当前代码段解析，可独立滚动"></div></section></div></section>';
  $('#file-view').innerHTML=html;
  $$('[data-file]').forEach(a=>a.classList.toggle('active',a.dataset.file===path));
  $$('[data-select-segment]').forEach(button=>button.onclick=()=>openSegment(Number(button.dataset.selectSegment),{scrollSource:false}));
  $$('.source-segment pre').forEach(pre=>pre.onclick=()=>{if(!getSelection()?.toString())openSegment(Number(pre.closest('[data-segment]').dataset.segment),{scrollSource:false})});
  $$('[data-open-line]').forEach(button=>button.onclick=()=>{openSegment(f.segments.findIndex(s=>s.start===Number(button.dataset.openLine)));revealWorkbench()});
  $('#segment-picker').onchange=e=>openSegment(Number(e.target.value));
  $('#previous-segment').onclick=()=>openSegment(currentSegment-1);
  $('#next-segment').onclick=()=>openSegment(currentSegment+1);
  $('#reader-focus').onclick=()=>{document.body.classList.toggle('reader-wide');$('#reader-focus').textContent=document.body.classList.contains('reader-wide')?'显示文件目录':'展开阅读区';revealWorkbench()};
  $('#copy-file').onclick=()=>copy(f.source);$('#download-file').onclick=()=>download(f.source,f.path.split('/').pop(),'text/plain;charset=utf-8');
}
function stepRange(step,s){
  const m=(step.title+' '+step.detail).match(/L(\d+)(?:[–—−-](\d+))?/);
  return m?{start:Math.max(s.start,Number(m[1])),end:Math.min(s.end,Number(m[2]||m[1]))}:null;
}
function highlightSourceLines(start,end){
  $$('#source-scroll .code-line').forEach(line=>line.classList.toggle('step-highlight',Number(line.dataset.codeLine)>=start&&Number(line.dataset.codeLine)<=end));
  const target=$('#source-scroll [data-code-line="'+start+'"]'),scroller=$('#source-scroll');
  if(target)scroller.scrollTop+=target.getBoundingClientRect().top-scroller.getBoundingClientRect().top-40;
}
function openSegment(index,options={}){
  const f=currentFile,s=f?.segments[index];if(!s)return;
  currentSegment=index;
  $$('.source-segment').forEach(el=>el.classList.toggle('selected',Number(el.dataset.segment)===index));
  $$('[data-select-segment]').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.selectSegment)===index)));
  $$('#source-scroll .step-highlight').forEach(el=>el.classList.remove('step-highlight'));
  $('#segment-picker').value=String(index);
  $('#selection-status').textContent=(index+1)+' / '+f.segments.length+' · L'+s.start+'–'+s.end;
  $('#previous-segment').disabled=index===0;$('#next-segment').disabled=index===f.segments.length-1;
  const visuals=s.visuals||[],previous=f.segments[index-1],next=f.segments[index+1];
  const steps=s.steps.map(step=>{const range=stepRange(step,s);return '<li>'+(range&&range.end>=range.start?'<button class="step-source-link" data-step-start="'+range.start+'" data-step-end="'+range.end+'"><strong>'+esc(step.title)+'</strong><span>定位源码</span></button>':'<strong>'+esc(step.title)+'</strong>')+'<p>'+esc(step.detail)+'</p></li>'}).join('');
  $('#analysis-content').innerHTML='<div class="analysis-intro"><span class="reader-kicker">当前代码段 · L'+s.start+'–'+s.end+'</span>'+coreBadge(s)+'<h2 id="analysis-title">'+esc(s.title)+'</h2><p class="analysis-summary">'+esc(s.summary)+'</p>'+(previous?'<p class="segment-context"><span>前一段</span>'+esc(previous.title)+'</p>':'<p class="segment-context">从本文件的定义与准备工作开始阅读。</p>')+(s.shapes?'<div class="shape">'+esc(s.shapes)+'</div>':'')+'</div><nav class="analysis-jumps" aria-label="当前解析内容">'+(visuals.length?'<button data-analysis-anchor="analysis-visuals">图解</button>':'')+'<button data-analysis-anchor="analysis-steps">逐步执行</button><button data-analysis-anchor="analysis-data">参数与变量</button><button data-analysis-anchor="analysis-syntax">语法细节</button></nav><section class="analysis-section" id="analysis-purpose"><h3>这段代码解决什么问题</h3>'+paragraphs(s.principle)+'</section>'+(visuals.length?'<section class="analysis-section" id="analysis-visuals"><h3>把代码中的变化画出来</h3>'+visuals.map(visualFigure).join('')+'</section>':'')+'<section class="analysis-section" id="analysis-steps"><h3>对着左侧源码，逐步执行</h3><ol class="execution-steps">'+steps+'</ol></section><section class="analysis-section" id="analysis-data"><h3>参数与数据对照</h3><h4>调用时传入什么</h4>'+parameterTable(s.parameters)+'<h4>执行中产生和使用什么</h4>'+variableTable(s.variables)+'</section><section class="analysis-section" id="analysis-syntax"><h3>调用与语法怎样理解</h3>'+paragraphs(s.syntax)+'</section><section class="analysis-section"><h3>这一段交出了什么</h3>'+readerList(s.returns)+'</section>'+(s.pitfall?'<aside class="note warn"><strong>对照当前实现时要注意</strong>'+paragraphs(s.pitfall)+'</aside>':'')+(s.flow?'<details class="control-flow-details"><summary>补充：查看条件判断与控制分支</summary>'+flowFigure(s.flow)+'</details>':'')+(next?'<div class="next-context"><span>接下来</span><strong>'+esc(next.title)+'</strong><p>'+esc(next.summary)+'</p><button id="continue-segment">继续读下一段</button></div>':'<div class="next-context"><strong>本文件读到这里</strong><p>可以返回全局导读，继续下一站。</p><a href="'+codeHref('guide')+'">查看建议阅读顺序</a></div>');
  $('#analysis-content').scrollTop=0;
  $$('[data-analysis-anchor]').forEach(button=>button.onclick=()=>{const target=$('#'+button.dataset.analysisAnchor),scroll=$('#analysis-content');scroll.scrollTop+=target.getBoundingClientRect().top-scroll.getBoundingClientRect().top-16});
  $$('[data-step-start]').forEach(button=>button.onclick=()=>highlightSourceLines(Number(button.dataset.stepStart),Number(button.dataset.stepEnd)));
  if($('#continue-segment'))$('#continue-segment').onclick=()=>openSegment(index+1);
  if(options.scrollSource!==false){const target=$('#seg-'+s.start),scroll=$('#source-scroll');scroll.scrollTop+=target.getBoundingClientRect().top-scroll.getBoundingClientRect().top-10}
  if(options.updateURL!==false)history.replaceState(null,'',codeHref(f.path)+'?line='+s.start);
  if(options.reveal!==false)revealWorkbench();
}
document.addEventListener('click',event=>{
  const visualButton=event.target.closest('[data-visual-zoom]');
  if(visualButton){const visual=DATA.visuals.find(v=>v.id===visualButton.dataset.visualZoom);if(!visual)return;$('#figure-title').textContent=visual.title;$('#figure-image').src=visual.uri;$('#figure-image').alt=visual.alt;figureWidth=Number(visual.width)}
  else {const button=event.target.closest('[data-flow-zoom]');if(!button)return;const figure=button.closest('.reader-flow'),svg=figure.querySelector('svg');$('#figure-title').textContent=figure.querySelector('strong').textContent;$('#figure-image').src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg.outerHTML);$('#figure-image').alt=figure.querySelector('strong').textContent;figureWidth=Number(svg.getAttribute('width'))}
  $('#figure-dialog').showModal();setFigureScale(Math.min(1,($('#figure-stage').clientWidth-32)/figureWidth));$('#figure-stage').scrollTop=0;$('#figure-stage').scrollLeft=0;
});
