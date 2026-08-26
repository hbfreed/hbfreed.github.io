// Hover/focus popups for kramdown footnotes. Clones the footnote body from the
// list at the bottom of the page, so no content is duplicated in the markdown.
(function () {
  var popup = null;
  var hideTimer = null;

  function hide() {
    if (popup) { popup.remove(); popup = null; }
  }

  function show(link) {
    var id = decodeURIComponent((link.getAttribute('href') || '').replace(/^#/, ''));
    var note = id && document.getElementById(id);
    if (!note) return;
    hide();
    popup = document.createElement('div');
    popup.className = 'footnote-popup';
    popup.innerHTML = note.innerHTML;
    popup.querySelectorAll('.reversefootnote').forEach(function (a) { a.remove(); });
    popup.addEventListener('mouseenter', function () { clearTimeout(hideTimer); });
    popup.addEventListener('mouseleave', scheduleHide);
    document.body.appendChild(popup);

    var r = link.getBoundingClientRect();
    var pw = popup.offsetWidth, ph = popup.offsetHeight;
    var left = Math.max(8, Math.min(r.left + window.scrollX, window.scrollX + document.documentElement.clientWidth - pw - 8));
    var below = r.bottom + 8 + ph < window.innerHeight;
    var top = below ? r.bottom + window.scrollY + 8 : r.top + window.scrollY - ph - 8;
    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 150);
  }

  document.querySelectorAll('a.footnote').forEach(function (link) {
    link.addEventListener('mouseenter', function () { clearTimeout(hideTimer); show(link); });
    link.addEventListener('mouseleave', scheduleHide);
    link.addEventListener('focus', function () { show(link); });
    link.addEventListener('blur', scheduleHide);
  });

  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });
})();
