/* Source reader. Embedded inside app.js's closure by build.py. */
let graphSerial=0;
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
  return '<figure class="reader-flow"><div class="reader-flow-head"><strong>'+esc(graph.title)+'</strong><button type="button" data-flow-zoom>放大流程图</button></div><div class="reader-flow-canvas" tabindex="0" role="region" aria-label="'+esc(graph.title)+'，可滚动查看">'+graphSVG(graph)+'</div><figcaption>'+esc(graph.caption||'按箭头阅读数据与控制的流向。')+'<span>长图可滚动查看，也可放大阅读。</span></figcaption></figure>';
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
function renderFile(path){
  const f=files.find(f=>f.path===path);currentFile=f||null;
  if(!f){$('#file-view').innerHTML='<h2>未找到这个文件</h2><p>请从左侧文件树选择一个文件。</p>';return}
  let html='<article class="file-guide"><div class="file-heading"><div><span class="reader-kicker">文件导读</span><h2>'+esc(f.path)+'</h2></div><div class="source-actions"><button id="copy-file">复制源码</button><button id="download-file">下载文件</button></div></div><div class="file-purpose"><div class="file-role">'+esc(f.role)+'</div><h3>'+esc(f.title)+'</h3>'+paragraphs(f.overview)+'</div><div class="file-contract"><section><h3>接收什么</h3>'+readerList(f.inputs)+'</section><section><h3>产生什么</h3>'+readerList(f.outputs)+'</section></div><section class="file-components"><h3>通过哪些部分完成工作</h3>'+componentTable(f)+'</section>'+flowFigure(f.pipeline)+'<div class="file-reading"><h3>放回项目中理解</h3>'+readerList(f.readingGuide)+'</div>';
  html+=typeGuide(f);
  if(f.dependencies?.length)html+='<div class="file-dependencies"><span>直接依赖的项目文件</span>'+f.dependencies.map(p=>'<a href="#code/'+encodeURI(p)+'">'+esc(p)+'</a>').join('')+'</div>';
  if(f.definitions?.length)html+='<details class="definition-index"><summary>全部函数与类的定义索引 · '+f.definitions.length+' 项</summary><div>'+f.definitions.map(d=>'<button data-open-line="'+d.segmentStart+'"><code>'+esc(d.signature)+'</code><span>L'+d.line+' · '+esc(d.scope||'文件级')+'</span></button>').join('')+'</div></details>';
  html+='</article><div class="reader-source-heading"><div><h2>逐段阅读源码</h2><p>展开任一片段，查看完整参数、变量、执行步骤与原理。</p></div><span>'+f.lines.length+' 行 · '+f.segments.length+' 段</span></div><nav class="segment-index" aria-label="本文件讲解目录">'+f.segments.map((s,i)=>'<button data-scroll-line="'+s.start+'"><span>L'+s.start+'</span>'+esc(s.title)+'</button>').join('')+'</nav>';
  let cursor=1;
  for(let i=0;i<f.segments.length;i++){
    const s=f.segments[i];
    if(s.start>cursor)html+='<details class="segment gap"><summary>L'+cursor+'–'+(s.start-1)+' · 源码上下文</summary>'+codeBlock(f,cursor,s.start-1)+'</details>';
    html+='<article class="segment" id="seg-'+s.start+'"><div class="segment-head"><div><small>L'+s.start+'–'+s.end+'</small><h3>'+esc(s.title)+'</h3></div><button data-explain="'+i+'">展开讲解</button></div><p class="segment-summary">'+esc(s.summary)+'</p>'+codeBlock(f,s.start,s.end)+'</article>';
    cursor=s.end+1;
  }
  if(cursor<=f.lines.length)html+='<details class="segment gap"><summary>L'+cursor+'–'+f.lines.length+' · 剩余完整源码</summary>'+codeBlock(f,cursor,f.lines.length)+'</details>';
  $('#file-view').innerHTML=html;
  $$('[data-file]').forEach(a=>a.classList.toggle('active',a.dataset.file===path));
  $$('[data-explain]').forEach(b=>b.onclick=()=>openSegment(Number(b.dataset.explain)));
  $$('[data-open-line]').forEach(b=>b.onclick=()=>openSegment(f.segments.findIndex(s=>s.start===Number(b.dataset.openLine))));
  $$('[data-scroll-line]').forEach(b=>b.onclick=()=>$('#seg-'+b.dataset.scrollLine)?.scrollIntoView({block:'start'}));
  $('#copy-file').onclick=()=>copy(f.source);
  $('#download-file').onclick=()=>download(f.source,f.path.split('/').pop(),'text/plain;charset=utf-8');
}
function openSegment(index){
  const f=currentFile,s=f?.segments[index];if(!s)return;
  if(!$('#explanation').open)returnFocus=document.activeElement;
  currentSegment=index;$('#drawer-path').textContent=f.path+' / L'+s.start+'–'+s.end;
  const parts=[['purpose','作用与原理'],['params','参数'],['variables','变量'],['steps','执行过程'],['syntax','语法与调用'],['returns','输出'],['source','源码']];
  $('#drawer-content').innerHTML='<div class="explain-intro"><span class="reader-kicker">源码详解 · '+(index+1)+' / '+f.segments.length+'</span><h2 id="drawer-title">'+esc(s.title)+'</h2><p>'+esc(s.summary)+'</p>'+(s.shapes?'<div class="shape">'+esc(s.shapes)+'</div>':'')+'</div><nav class="explain-nav" aria-label="讲解内容">'+parts.map(([key,label])=>'<button data-reader-anchor="explain-'+key+'">'+label+'</button>').join('')+'</nav><section id="explain-purpose" class="explain-part"><h3 tabindex="-1">作用与计算原理</h3>'+paragraphs(s.principle)+'</section><section id="explain-params" class="explain-part"><h3 tabindex="-1">函数参数与配置</h3>'+parameterTable(s.parameters)+'</section><section id="explain-variables" class="explain-part"><h3 tabindex="-1">重要变量与数据结构</h3>'+variableTable(s.variables)+'</section><section id="explain-steps" class="explain-part"><h3 tabindex="-1">内部怎样一步步执行</h3>'+flowFigure(s.flow)+'<ol class="execution-steps">'+s.steps.map(step=>'<li><strong>'+esc(step.title)+'</strong><p>'+esc(step.detail)+'</p></li>').join('')+'</ol></section><section id="explain-syntax" class="explain-part"><h3 tabindex="-1">语法、函数调用与具体参数</h3>'+paragraphs(s.syntax)+'</section><section id="explain-returns" class="explain-part"><h3 tabindex="-1">返回值与后续用途</h3>'+readerList(s.returns)+'</section>'+(s.pitfall?'<aside class="note warn"><strong>对照实际实现时要注意</strong>'+paragraphs(s.pitfall)+'</aside>':'')+'<section id="explain-source" class="explain-part"><h3 tabindex="-1">对照原始源码 · L'+s.start+'–'+s.end+'</h3>'+codeBlock(f,s.start,s.end)+'</section><div class="drawer-actions"><button id="previous-segment" '+(index===0?'disabled':'')+'>上一段</button><span>'+(index+1)+' / '+f.segments.length+'</span><button id="next-segment">'+(index<f.segments.length-1?'下一段':'返回文件')+'</button></div>';
  $('#previous-segment').onclick=()=>openSegment(index-1);
  $('#next-segment').onclick=()=>index<f.segments.length-1?openSegment(index+1):$('#explanation').close();
  $$('[data-reader-anchor]').forEach(b=>b.onclick=()=>{const target=$('#'+b.dataset.readerAnchor);target.scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});target.querySelector('h3').focus({preventScroll:true})});
  if(!$('#explanation').open)$('#explanation').showModal();
  $('#explanation').scrollTop=0;
}
document.addEventListener('click',event=>{
  const button=event.target.closest('[data-flow-zoom]');if(!button)return;
  const figure=button.closest('.reader-flow'),svg=figure.querySelector('svg');
  $('#figure-title').textContent=figure.querySelector('strong').textContent;
  $('#figure-image').src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg.outerHTML);
  $('#figure-image').alt=figure.querySelector('strong').textContent;
  figureWidth=Number(svg.getAttribute('width'));
  $('#figure-dialog').showModal();setFigureScale(Math.min(1,($('#figure-stage').clientWidth-32)/figureWidth));
  $('#figure-stage').scrollTop=0;$('#figure-stage').scrollLeft=0;
});
