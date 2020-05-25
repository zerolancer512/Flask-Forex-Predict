$(function(){
    $('button').click(function (){
        var Close = $('#Close').val();
        var High = $('#High').val();
        var Low = $('#Low').val();
        var return_1 = $('#return_1').val();
        var return_2 = $('#return_2').val();
        var return_3 = $('#return_3').val();
        $.ajax(
                { url: '/predict',
                data: $('predict').serialize(),
                type: "POST",
                success: function (response) {
                    console.log(response)
                    },
                error: function (error){
                    console.log(error);
                }
                 
            });
    
        });
}); 