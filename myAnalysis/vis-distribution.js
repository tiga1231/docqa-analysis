d3.selection.prototype.moveToFront = function() {  
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};
d3.selection.prototype.moveToBack = function() {  
    return this.each(function() { 
        var firstChild = this.parentNode.firstChild; 
        if (firstChild) { 
            this.parentNode.insertBefore(this, firstChild); 
        } 
    });
};


flatten = _.shuffle(flatten);

let svgWidth = window.innerWidth/3;
let svgHeight = window.innerHeight/2;
let marginMin = 30;
let width = svgWidth-2*marginMin;
let height = svgHeight-2*marginMin;
let side = Math.min(width, height);
let marginLeft = (svgWidth-side)/2;
let marginTop = (svgHeight-side)/2;

let body = d3.select('body');
let leftDiv = body.append('div')
.style('float', 'left');

let svg = leftDiv
.append('div')
.append('svg')
.attr('width', svgWidth)
.attr('height', svgHeight)
.style('background', '#fff');

let probBarSvg = leftDiv
.append('div')
.append('svg')
.attr('width', svgWidth)
.attr('height', svgHeight)
.style('background', '#fff');

probBarSvg.show = function(d){
	let preds = d.preds;
	let sx = d3.scaleLinear()
	.domain([0,preds.length+0.5])
	.range([marginLeft, svgWidth-marginLeft]);
	let sy = d3.scaleLinear()
	.domain([0, 1])
	.range([svgHeight-marginTop, marginTop]);
	let ax = d3.axisBottom(sx);
	let ay = d3.axisLeft(sy);
	let sc = d3.scaleOrdinal(d3.schemeCategory10);
	sc(0); sc(1);

	this.selectAll('.bar')
	.data(preds)
	.enter()
	.append('rect')
	.attr('class', 'bar');

	this.selectAll('.bar')
	.attr('x', (d,i)=>sx(i+0.6))
	.attr('y', (d,i)=>sy(d.prob))
	.attr('width', Math.abs(sx(1)-sx(0))*0.8)
	.attr('height', (d,i)=>Math.abs(sy(d.prob)-sy(0)))
	.attr('fill', (_,i)=>{
		return d.true_pred_index==i?sc(1):sc(0)
	});

	this.selectAll('.x-axis')
	.data([0])
	.enter()
	.append('g')
	.attr('class', 'x-axis')
	.attr('transform', 'translate('+0+','+sy(0)+')')
	this.selectAll('.x-axis')
	.call(ax);

	this.selectAll('.y-axis')
	.data([0])
	.enter()
	.append('g')
	.attr('class', 'y-axis')
	.attr('transform', 'translate('+sx(0)+','+0+')')
	this.selectAll('.y-axis')
	.call(ay);
};	







let sx = d3.scaleLinear()
.domain([0,1])
.range([marginLeft, svgWidth-marginLeft]);
let sy = d3.scaleLinear()
.domain([-0.5, 0.5])
.range([svgHeight-marginTop, marginTop]);
let ax = d3.axisBottom(sx);
let ay = d3.axisLeft(sy);

let sc = d3.scaleOrdinal(d3.schemeCategory10);
sc(true);
sc(false);

// let sc = d3.scaleOrdinal([d3.schemeCategory10[0]]);

let tip = svg.append('text')
.attr('class', 'tip')
.style('background-color', '#eee')
.style('pointer-events', 'none');

tip.show = function(d){
	this.attr('x', sx(d.top_pred_prob))
	.attr('y', sy(d.true_pred_prob))
	.text(d.title);
	this.moveToFront();
}


let currentD = null;
let currentPoint = null;

svg.selectAll('.point')
.data(flatten)
.enter()
.append('circle')
.attr('class', 'point')
.attr('cx', d=>sx(d.top_pred_prob))
.attr('cy', d=>sy(d.true_pred_prob))
.attr('r', 2)
.attr('fill', d=>sc(!d.is_impossible))
.attr('fill-opacity', 0.4)
.on('mouseover', function(d,i){

	svg.selectAll('.point')
	.attr('stroke-width', 0);

	d3.select(this)
	.attr('stroke', 'white')
	.attr('stroke-width', 10);

	d3.select(this).moveToFront();
	tip.show(d);
	probBarSvg.show(d);
	show(d);
})
.on('mouseout', function(){
	if(currentD){
		show(currentD);
		probBarSvg.show(currentD);
		tip.show(currentD);
	}
	

	svg.selectAll('.point')
	.attr('stroke-width', 0);
	if(currentPoint){
		currentPoint
		.attr('stroke-width',10);
	}
	

})
.on('click', function(d,i){
	currentPoint = d3.select(this);
	currentD = d;
});

svg.append('g')
.attr('class', 'x-axis')
.attr('transform', 'translate('+0+','+sy(0)+')')
.call(ax);

svg.append('g')
.attr('class', 'y-axis')
.attr('transform', 'translate('+sx(0)+','+0+')')
.call(ay);

svg.append('text')
.attr('class', 'xlabel')
.attr('x', sx(0.5))
.attr('y', sy(-0.25))
.attr('text-anchor', 'middle')
.text('Top prediction probability');

svg.append('text')
.attr('class', 'ylabel')
.attr('x', marginLeft/3)
.attr('y', sy(0.0))
.attr('transform', 'rotate(-90,'+(marginLeft/3)+','+sy(0.0)+')')
.attr('text-anchor', 'middle')
.text('True prediction probability');
















let div;
function show(d){
	body.selectAll('#paragraph-container').remove();
	if(d.answers.length==0){
	d.answers = [{text:'', answer_start:0}];
}

	let scPred = d3.scaleLinear()
	.domain([0, d3.max(d.preds, (e)=>e.prob)])
	.range(['#eee','#9ecae1','#3182bd']);
	let scAns = d3.scaleLinear()
	.domain([0, d3.max(d.preds, (e)=>e.prob)])
	.range(['#eee','#fdbb84','#e34a33']);

	div = body.append('div')
	.attr('id', 'paragraph-container')
	.style('width', '60%')
	.style('margin-left', '50px')
	.style('float', 'left');

	div.append('h1')
	.attr('class', 'title')
	.text(d.title);

	div.append('p')
	.attr('class', 'context')
	.html(taggedHtml(d.context, d.answers, d.preds));

	div.append('h2')
	.text('Question');

	div.append('p')
	.attr('class', 'question')
	.text(d.question);

	div.append('p')
	.attr('class', 'is_impossible')
	.text('possible? ' + !d.is_impossible);


	// TRUE ANSWERS
	div.append('h2')
	.text('True answers')
	.on('mouseover', function(){
		d3.select(this)
		.style('background-color', '#eee');
		showDistribution('yAns', scAns);

	})
	.on('mouseout', function(){
		d3.select(this)
		.style('background-color', '');
		hideDistribution()
	});

	div.append('ol')
	.attr('class', 'answers')
	.selectAll('li')
	.data(d.answers)
	.enter()
	.append('li')
	.attr('class', 'answer')
	.text(d=>d.text==''? "<NO ANSWER>" : d.text)
	.on('mouseover', d=>{
		highlight([d.answer_start, d.answer_start+d.text.length]);
	})
	.on('mouseout', d=>{
		dehighlight([d.answer_start, d.answer_start+d.text.length]);
	});



	// PREDICTED ANSWERS
	div.append('h2')
	.text('Predicted answers')
	.on('mouseover', function(){
		d3.select(this)
		.style('background-color', '#eee');
		showDistribution('yPred', scPred);

	})
	.on('mouseout', function(){
		d3.select(this)
		.style('background-color', '');
		hideDistribution();
	});

	div.append('ol')
	.attr('class', 'preds')
	.selectAll('li')
	.data(d.preds)
	.enter()
	.append('li')
	.attr('class', 'pred')
	.style('color', e=>{
		let answerSet = new Set(d.answers.map(x=>x.text));
		if(answerSet.has(e.answer)){
			return sc(1);
		}else{
			return 'black';
		}
	})
	.text(d=> d.answer=='' ? "<NO ANSWER>":d.answer)
	.on('mouseover', d=>{
		highlight(d.span);
	})
	.on('mouseout', d=>{
		dehighlight(d.span);
	});

}

function showDistribution(attr, sc){
	div.select('.context')
	.selectAll('span')
	.style('background-color', function(){
		let y = +d3.select(this).attr(attr);
		return sc(y);
	});
}


function hideDistribution(){
	div.select('.context')
	.selectAll('span')
	.style('background-color', '');
}


function highlight(span){
	let selection;
	if(span[0]==0 && span[1]==0){
		selection = div.select('.context')
		.select('span#noAnswer')
	}else{
		selection = div.select('.context')
		.selectAll('span')
		.filter(function(){
			let i = +d3.select(this).attr('i');
			return i>=span[0] && i<span[1];
		})
	}
	selection.style('color', 'red')
	.style('text-decoration','underline');	
}


function dehighlight(span){
	let selection;
	if(span[0]==0 && span[1]==0){
		selection = div.select('.context')
		.select('span#noAnswer')
	}else{
		selection = div.select('.context')
		.selectAll('span')
		.filter(function(){
			let i = +d3.select(this).attr('i');
			return i>=span[0] && i<span[1];
		});
	}
	selection.style('color', '')
	.style('text-decoration','');
}


function zeros(length){
	return Array.from(Array(length), ()=>0);
}

function taggedHtml(text, answers, preds){
	let res = '';
	let yAnswer = answer2y(answers, text.length);
	let yPred = pred2y(preds, text.length);


	for(let i=0; i<text.length; i++){
		res += '<span yPred='+yPred[i]
		+' yAns='+yAnswer[i]
		+' i='+i
		+'>';
		res += text[i];
		res += '</span>';
	}

	let yPred_no;
	if(preds.filter(d=>d.answer.length==0).length > 0){
		yPred_no = preds.filter(d=>d.answer.length==0)[0].prob;
	}else{
		yPred_no = 0.0;
	}
	let y_no = (answers.length==1 && answers[0].text=='')? 1.0:0.0;
	res += '<span id="noAnswer" yPred='+yPred_no+' yAns='+y_no+'>'
	res += 'NO ANSWER';
	res += '</span>';
	
	return res;
}

function answer2y(answers, length){
	let res = zeros(length);
	for(let a of answers){
		for(let i=a.answer_start; i<a.answer_start+a.text.length; i++){
			res[i]+=1;
		}
	}
	return res;
}

function pred2y(preds, length){
	let res = zeros(length);
	for(let p of preds){
		for(let i=p.span[0]; i<p.span[1]; i++){
			res[i]+=p.prob;
		}
	}
	return res;
}