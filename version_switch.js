// Modified from the cpython source found in Doc/tools/sphinxext/static/version_switch.js
// which is copyright 2013 PSF and Licensed under the PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2

(function() {

  var package = "eofs";
  var url_re = new RegExp('/' + package + '/(latest|(v\\d+\\.\\d+)|dev)/');                                     

  function patch_url(url, new_version) {
    function replacer(match, part1, offset, string) {
        return match[0].replace(part1, new_version);
    }
    var new_url = url.replace(url_re, '/' + package + '/' + new_version + '/');
    return new_url;
  }

  function on_switch() {
    var selected = $(this).children('option:selected').attr('value');

    var url = window.location.href,
        new_url = patch_url(url, selected);

    if (new_url != url) {
      // check beforehand if url exists, else redirect to version's start page
      $.ajax({
        url: new_url,
        success: function() {
           window.location.href = new_url;
        },
        error: function() {
           window.location.href = '/' + package + '/' + selected;
        }
      });
    }
  }

  $(document).ready(function() {  
    var version_heading = $('div.sphinxsidebarwrapper').find("h2");
    version_heading.after('<div class="version_switcher right"><select></select></div>');
    var version_switcher = $('.version_switcher select');
    version_switcher.bind('change', on_switch);
    var url = window.location.href;
    var match = url_re.exec(url);
    var current_version = match[1];
  
    $.getJSON('/' + package + '/versions.json', function(data) {
      $.each(data, function(index, version) {
          var option = '<option value="' + version + '"';
          if (version == current_version)
              option += ' selected="selected">' + version + '</option>';
          else
            option += '>' + version + '</option>';
          version_switcher.append(option);
      });
    });

  });

})();
