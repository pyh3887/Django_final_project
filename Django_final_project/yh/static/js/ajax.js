
$(document).ready(function(){
	
	$('#Progress_Loading').hide(); //첫 시작시 로딩바를 숨겨준다.
	$('#Progress_Loading2').hide();
	
	$.ajax({
		url:'show3',
		type:'get',
		dataType:'json',
		async:true,
		success:function(data){
			//alert(data)
			
			maegraph = data['ks_graph_mae']
			msegraph = data['ks_graph_mse']
			lineargraph = data['ks_graph_linear']
			graph1 = data['ks_graph1']
			graph2 = data['ks_graph2']
			loss = data['loss']
			mae = data['mae']
			mse = data['mse']
			acc = data['acc']
			acc = data['acc']
			acc = data['acc']
			
			$('#maegraph').append(maegraph)	
			$('#msegraph').append(msegraph)	
			$('#lineargraph').append(lineargraph)	
			$('#graph1').append(graph1)	
			$('#graph2').append(graph2)	
			//$('#loss').append(loss)	
			//$('#mae').append(mae)	
			//$('#mse').append(mse)	
			//$('#acc').append(acc)	
				
		},
		error:function(){
			$('#showData').text('error')
		}
	})
	//alert("a")
	//버거를 눌렀을 시
	$("#burger").bind('click',function(){
		$('#Progress_Loading').show(); //ajax실행시 로딩바를 보여준다.
		$("#showData").empty()
		//alert('b')
			$.ajax({
				url:'show',
				type:'get',
				dataType:'json',
				async:true,
				success:function(data){
					//alert(data)
					alert(data['a'])
					str1 = "<h1>"+ data['a'] + "</h1>"
					str1 += "<h1>정확도 :"+ data['accuracy'] + "%</h1>"
					str2 = "<div>" + data['yh_grap3']+ "</div>" 					
					$('#showData').append(str1)	
					$('#showData2').append(str2)	
					$('#Progress_Loading').hide(); //ajax종료시 로딩바를 숨겨준다.		
				},
				error:function(){
					$('#showData').text('error')
				}
			})
		})
		
	$("#anal2").bind('click',function(){
		$('#Progress_Loading2').show(); //ajax실행시 로딩바를 보여준다.
		$("#showData").empty()
		//alert('b')
			$.ajax({
				url:'show2', 
				type:'get',
				dataType:'json',
				async:true,
				success:function(data){
					//alert(data)
					alert(data['anal'])
					str1 = "<h1>"+ data['anal'] + "</h1>"					
					$('#showData4').append(str1)
					$('#Progress_Loading2').hide(); //ajax종료시 로딩바를 숨겨준다.					
				
				},
				error:function(){
					$('#showData').text('error')
				}
			})
		})

})
