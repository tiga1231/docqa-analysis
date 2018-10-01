// var dataset loaded from dataset.js
let data = dataset.data;

let articles = data[_.random(0, data.length-1)];

let title = articles.title;

let paragraphs = articles.paragraphs;
let paragraph = paragraphs[_.random(0, paragraphs.length-1)];

let context = paragraph.context;

let qas = paragraph.qas;
let qa = qas[_.random(0,qas.length-1)];

let question = qa.question;
let answers = qa.answers;
if(answers.length==0){
	answers = [{text:'', answer_start:0}];
}
let preds = qa.preds;
let is_impossible = qa.is_impossible;

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
	console.log(y_no);
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



let body = d3.select('body');
let rootDiv = body.append('div')
.attr('class', 'rootDiv');

let scPred = d3.scaleLinear()
.domain([0, d3.max(preds, (d)=>d.prob)])
.range(['#eee','#9ecae1','#3182bd']);
let scAns = d3.scaleLinear()
.domain([0, d3.max(preds, (d)=>d.prob)])
.range(['#eee','#fdbb84','#e34a33']);


rootDiv.append('h1')
.attr('class', 'title')
.text(title);

rootDiv.append('p')
.attr('class', 'context')
.html(taggedHtml(context, answers, preds));

rootDiv.append('h2')
.text('Question');

rootDiv.append('p')
.attr('class', 'question')
.text(question);

rootDiv.append('p')
.attr('class', 'is_impossible')
.text('possible? ' + !is_impossible);










// TRUE ANSWERS
rootDiv.append('h2')
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

rootDiv.append('ol')
.attr('class', 'answers')
.selectAll('li')
.data(answers)
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
rootDiv.append('h2')
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

rootDiv.append('ol')
.attr('class', 'preds')
.selectAll('li')
.data(preds)
.enter()
.append('li')
.attr('class', 'pred')
.text(d=> d.answer=='' ? "<NO ANSWER>":d.answer)
.on('mouseover', d=>{
	highlight(d.span);
})
.on('mouseout', d=>{
	dehighlight(d.span);
})









function showDistribution(attr, sc){
	rootDiv.select('.context')
	.selectAll('span')
	.style('background-color', function(){
		let y = +d3.select(this).attr(attr);
		return sc(y);
	});
}


function hideDistribution(){
	rootDiv.select('.context')
	.selectAll('span')
	.style('background-color', '');
}


function highlight(span){
	let selection;
	if(span[0]==0 && span[1]==0){
		selection = rootDiv.select('.context')
		.select('span#noAnswer')
	}else{
		selection = rootDiv.select('.context')
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
		selection = rootDiv.select('.context')
		.select('span#noAnswer')
	}else{
		selection = rootDiv.select('.context')
		.selectAll('span')
		.filter(function(){
			let i = +d3.select(this).attr('i');
			return i>=span[0] && i<span[1];
		});
	}
	selection.style('color', '')
	.style('text-decoration','');
}






