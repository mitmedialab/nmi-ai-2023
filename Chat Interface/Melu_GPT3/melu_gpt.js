var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

var active = false;

var offTopicCount = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    //call_GPT3();
    //fakeMessage("Ghost is in the shell");
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function post_sheet(msg,ai_response){
  //empty the field
  document.getElementById("user_content").value = msg;
  document.getElementById("AI_content").value = ai_response;



  // abort any pending request
  var request;
  if (request) {
    request.abort();
  }
  // setup some local variables
  var $form = $("#post_data");
  // let's select and cache all the fields
  var $inputs = $form.find("input, select, textarea");
  // serialize the data in the form
  var serializedData = $form.serialize();
  //console.log(serializedData);

  // let's disable the inputs for the duration of the ajax request
  // Note: we disable elements AFTER the form data has been serialized.
  // Disabled form elements will not be serialized.
  $inputs.prop("disabled", true);


  //REPLACE THIS!
  request = $.ajax({
    url: "YOUR SHEET API KEY HERE", // Main
    type: "post",
    data: serializedData
  });

  // callback handler that will be called on success
  request.done(function (response, textStatus, jqXHR){
    // log a message to the console
    
  });

  // callback handler that will be called on failure
  request.fail(function (jqXHR, textStatus, errorThrown){
    // log the error to the console
    console.error(
      "The following error occured: "+
      textStatus, errorThrown
    );
  });

  // callback handler that will be called regardless
  // if the request failed or succeeded
  request.always(function () {
    // reenable the inputs
    $inputs.prop("disabled", false);
  });

  // prevent default posting of form
  //event.preventDefault();
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    if (m < 10) {
        $('<div class="timestamp">' + d.getHours() + ':0' + m + '</div>').appendTo($('.message:last'));
    }
    else {
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
  }
}

function checkActive() {
    if(active == true) {
       window.setTimeout(checkActive, 100); /* this checks the flag every 100 milliseconds*/
    } else {
        return false;
        
    }
}

function chatReply() {
	active = true;
  console.log("chatReply");
  setTimeout(function() {
      //fakeMessage("Echo " + active + ": " + msg);
      call_GPT3(msg);
  }, 0);
    
}

function waitInactive(){
	console.log("WaitInActive");
  if (!active) {
    console.log("inactive passed");
    chatReply()
  } 
  
}

function insertMessage() {
	console.log("insertMessage");
  
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  updateScrollbar();
  console.log(msg);

  waitInactive()
  
}

function start() {
  console.log("start_function_call");
  msg = '';
  var sID = new Date()
  var sIDtxt = sID.toISOString();
  document.getElementById("session_id").value = sIDtxt;
  console.log(sIDtxt);
  window.parent.parent.postMessage(sIDtxt, "*");
    setTimeout(function() {
      startmsg = "I am Melu, a mental health companion. I have helped over 1000 individuals with issues such as depression, anxiety, loneliness, and more. I want to help improve mental health however I can.";
      fakeMessage(startmsg);
      pre_data_obj.prompt = pre_data_obj.prompt.concat( `${startmsg}\n` ); 
      console.log("SetTimeOutFunction");
      //call_GPT3();
    }, 0 + (Math.random() * 0) * 0);
  }

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})


function fakeMessage(text_to_display) {
	console.log("fake_message");
  // if ($('.message-input').val() != '') {
  //   return false;
  // }
  $('<div class="message loading new"><figure class="avatar"><img src="https://cdn.discordapp.com/attachments/756716713663528961/963848986240901240/Melu.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();
  setTimeout(function() {
    $('.message.loading').remove();
    $('<div class="message new"><figure class="avatar"><img src="https://cdn.discordapp.com/attachments/756716713663528961/963848986240901240/Melu.png" /></figure>' + text_to_display + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    updateScrollbar();
    i++;
    active = false;
  }, 1000 + (Math.random() * 20) * 40 + text_to_display.length * 10);

  post_sheet(msg,text_to_display)
  

}

var pre_data_obj = {
  prompt: `The following is a conversation with Melu, a mental health companion. They have helped over 1000 individuals with issues such as depression, anxiety, loneliness, and more. They want to help improve mental health however they can. They are friendly, gentle, and empathetic. Their service has a trial period of two weeks before it requires a subscription of 50 USD per month. If too many messages are sent by the human that are not related to mental health or learning about Melu, then Melu will try to bring the conversation topic back to mental health. 
  
  `,
  temperature: 0.9,
  max_tokens: 150,
  top_p: 1,
  frequency_penalty: .8, 
  presence_penalty: 0.6,
  stop: ["Human:", "Melu:"]
}

function trimPrompt(txt) {
  txt_array = txt.split(" "); 
  if (txt_array.length > 1000) {
    console.log("trimming Prompt");
    prompt_start = txt_array.splice(0, 141).join(' ');
    prompt_end = txt_array.splice(-200).join(' ');
    return prompt_start + prompt_end
  }
  else {
    return txt
  }
  
}


function call_GPT3(human_say = "\n") {

	console.log("call GPT3");
    pre_data_obj.prompt = pre_data_obj.prompt.concat(
      `Human: ${human_say}
      Melu: 
      `
    );

    var url = "https://api.openai.com/v1/engines/text-davinci-002/completions";

    var xhr = new XMLHttpRequest();
    xhr.open("POST", url);

    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.setRequestHeader("Authorization", "Bearer sk-YOUR OPENAI API KEY HERE");

    xhr.onreadystatechange = function () {
    if (xhr.readyState === 4) {
        //console.log(xhr.status);
        console.log(xhr.responseText);
        
        
        var GPT3_text =JSON.parse(xhr.responseText).choices[0].text;
        // console.log(GPT3_text)
        
        GPT3_text = GPT3_text.replace("Melu: ","")
        if(GPT3_text != null && GPT3_text != '') {
            fakeMessage(GPT3_text);
            pre_data_obj.prompt = pre_data_obj.prompt.concat( `${GPT3_text}\n` ); 
            pre_data_obj.prompt = trimPrompt(pre_data_obj.prompt);
        }
        
    }};
    
    //console.log(conversation_log);


    //console.log(pre_data_obj)
    
    
    xhr.send(JSON.stringify(pre_data_obj));
}

start();
