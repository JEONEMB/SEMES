'use strict';
document.getElementById('login-form')?.addEventListener('submit',async event=>{
  event.preventDefault();const button=event.target.querySelector('button');button.disabled=true;
  try{await UI.api('/api/admin/login','POST',{password:UI.$('password').value});location.href='/admin';}
  catch(error){UI.text('login-error',error.message);}
  finally{button.disabled=false;}
});
