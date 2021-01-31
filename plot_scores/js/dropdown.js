var original_chart=$(".main_block");
$.get('all_dir_names.txt', function(data) {
   console.log(data);
   all_dir = data.split("\n");
   $.each( all_dir , function( index, value ) {
        example_item = $(".example-dropdown-item");
        new_item = example_item.clone().removeClass("example-dropdown-item");
        new_item.text(value);
        score_file_name = "json/"+value+"/best_scores.json";
        zip_name = "submission/"+value+"_submission.zip";
        new_item.click({file_name:score_file_name, zip_name:zip_name}, get_score_file);
        new_item.appendTo( "#select-model-menu" );
    });
   
}, 'text');
