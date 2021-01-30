// Get data
function reset_chart(){
	$(".main_block").remove();
	original_chart.appendTo( "body");
}
function get_score_file(event) {
	reset_chart();
	$.getJSON(event.data.file_name, function(data) {
		all_scores = data['all_scores']
		plot_chart(all_scores);
	});
}

function plot_chart(all_scores) {
	var ctx = document.getElementById('myChart');
	// all metrics
	ALL_TASK_METRICS = {"mnli":["accuracy","mm_accuracy"], 
                    "rte":["accuracy"], 
                    "qqp":["accuracy","f1"], 
                    "qnli":["accuracy"], 
                    "mrpc":["accuracy","f1"], 
                    "sst2":["accuracy"], 
                    "cola":["matthews_correlation"],
                    "stsb":["pearson","spearmanr"]}
    ALL_TASK_COLORS = {"mnli":["rgba(200,0,0,1)","rgba(200,35,70,1)"], 
                    "rte":["rgb(209, 82, 38)"], 
                    "qqp":["rgb(209, 159, 38)","rgb(209, 188, 38)"], 
                    "qnli":["rgb(104, 226, 38)"], 
                    "mrpc":["rgb(9, 24, 224)","rgb(9, 175, 224)"], 
                    "sst2":["rgb(137, 228, 187)"], 
                    "cola":["rgb(158, 4, 234)"],
                    "stsb":["rgb(99, 95, 7)","rgb(179, 188, 7)"]}
	// Init dataset array
	datasets = new Array();
	for (var T = 0; T < Object.keys(ALL_TASK_METRICS).length; T++) {
		key = Object.keys(ALL_TASK_METRICS)[T];
		for (var M = 0; M < ALL_TASK_METRICS[key].length; M++) {
			m_scores = new Array();
			for (var i = 0; i < all_scores.length; i++) {
				m_scores.push(all_scores[i][T][M]);
			}
			dat = {
		    	type: 'line',
		    	label: 	key + "-" + ALL_TASK_METRICS[key][M],
		    	borderColor : ALL_TASK_COLORS[key][M],
		    	fill: false,
		    	data: m_scores
		    }
		    datasets.push(dat);
		}
	}
	var myChart = new Chart(ctx, {
	  type: 'line',
	  data: {
	    labels: Array.from({length: all_scores.length}, (_, i) => (i + 1).toString()),
	    datasets: datasets
	  }
	});
}

