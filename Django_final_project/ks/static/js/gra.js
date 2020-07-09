$(document).ready(function(){
	$("#gogo").bind('click', function(){
		$("#grap").empty()
		
			$.ajax({
	        url: "grap",
	        type: "GET",
			dataType:'json',
	        success: function (data) {
	            Plotly.newPlot('bargraph1', data.first_result,{});
	            Plotly.newPlot('bargraph2', data.second_result,{});
	            Plotly.newPlot('bargraph3', data.third_result,{});
	        },
			error:function(){
				$("#grap").text('에러났슈')
			}
		})
	})
})