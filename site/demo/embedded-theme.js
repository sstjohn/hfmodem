if(parent!==window)document.documentElement.dataset.deskEmbed='true';
/* Theme handshake only. This iframe preview does not connect to radio hardware. */
function applyDeskTheme(theme){document.documentElement.dataset.theme=theme==='dark'?'dark':'light'}
applyDeskTheme(new URLSearchParams(location.search).get('theme'));
window.addEventListener('message',event=>{if(event.origin!==location.origin||event.source!==parent)return;if(event.data?.kind==='desk-theme'){applyDeskTheme(event.data.theme);window.dispatchEvent(new Event('desk-theme-change'));}if(event.data?.kind==='desk-pause-review'){document.querySelectorAll('audio').forEach(a=>a.pause());window.dispatchEvent(new Event('desk-pause-review'));}});
function notifyDesk(data){if(parent!==window)parent.postMessage(data,location.origin)}
document.addEventListener('keydown',event=>{if(event.key==='Escape'){notifyDesk({kind:'stop-radio'});window.dispatchEvent(new Event('desk-pause-review'));}});
document.addEventListener('click',event=>{const a=event.target.closest('a');if(!a||parent===window)return;const url=new URL(a.href,location.href);if(url.origin!==location.origin)return;if(url.pathname.endsWith('/inspect.html')){event.preventDefault();notifyDesk({kind:'inspect-arm',arm:url.searchParams.get('arm')});}if(a.dataset.returnBook!==undefined){event.preventDefault();notifyDesk({kind:'back-book'});}});
