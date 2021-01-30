var original_chart=$(".main_block");
$.get('json/all_dir_names.txt', function(data) {
   console.log(data);
   all_dir = data.split(" ");
   $.each( all_dir , function( index, value ) {
        example_item = $(".example-dropdown-item");
        new_item = example_item.clone().removeClass("example-dropdown-item");
        new_item.text(value);
        score_file_name = "json/"+value+"/best_scores.json";
        new_item.click({file_name:score_file_name}, get_score_file);
        new_item.appendTo( "#select-model-menu" );
    });
   
}, 'text');
