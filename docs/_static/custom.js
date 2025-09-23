// Custom sidebar collapse/expand logic to avoid double click actions.
// We add a separate span.toggle so link navigation (a) and expansion are distinct.
document.addEventListener('DOMContentLoaded', function(){
  const menuRoot = document.querySelector('.wy-nav-side .wy-menu-vertical');
  if(!menuRoot) return;

  // Identify li elements that contain a nested UL.
  const parents = Array.from(menuRoot.querySelectorAll('li')).filter(li => li.querySelector(':scope > ul'));
  parents.forEach(li => {
    li.classList.add('has-children');
    // If anchor already has pseudo plus/minus we switch to a real element.
    const link = li.querySelector(':scope > a');
    if(!link) return;
    // Create toggle span
    const toggle = document.createElement('span');
    toggle.className = 'nav-toggle';
    toggle.setAttribute('role', 'button');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('tabindex', '0');
    toggle.innerText = '+';
    // Insert before link text
    link.prepend(toggle);

    // Start collapsed unless current page is inside
    const isCurrentPath = li.querySelector('a.current') || li.querySelector('li.current');
    if(isCurrentPath){
      li.classList.add('open');
      toggle.innerText = '–';
      toggle.setAttribute('aria-expanded','true');
    }

    // Hide toggle for items that are currently selected (no need to expand/collapse current page)
    if(li.querySelector(':scope > a.current')){
      toggle.style.display = 'none';
    }

    function toggleOpen(ev){
      ev.stopPropagation();
      ev.preventDefault();
      const open = li.classList.toggle('open');
      toggle.innerText = open ? '–' : '+';
      toggle.setAttribute('aria-expanded', open ? 'true':'false');
    }

    toggle.addEventListener('click', toggleOpen);
    toggle.addEventListener('keydown', (e)=>{
      if(e.key==='Enter' || e.key===' '){
        toggleOpen(e);
      }
    });

    // Prevent anchor itself from collapsing and navigating simultaneously.
    link.addEventListener('click', (e)=>{
      // If user clicked directly on toggle span, let toggle handler run; else normal nav.
      if(e.target === toggle){
        e.preventDefault();
      }
    }, true);
  });

  // Fix layout gaps and width issues
  function fixLayout() {
    // CRITICAL: Ensure content wrapper has proper margin to avoid overlap
    const contentWrap = document.querySelector('.wy-nav-content-wrap');
    if (contentWrap) {
      contentWrap.style.marginLeft = '240px';
    }

    // Fix the main content container
    const navContent = document.querySelector('.wy-nav-content');
    if (navContent) {
      navContent.style.maxWidth = 'none';
      navContent.style.padding = '1.618em 2em';
      navContent.style.margin = '0';
    }

    // Fix rst-content container
    const rstContent = document.querySelector('.wy-nav-content .rst-content');
    if (rstContent) {
      rstContent.style.maxWidth = 'none';
      rstContent.style.margin = '0';
    }

    // Fix document container
    const document = document.querySelector('.rst-content .document');
    if (document) {
      document.style.maxWidth = 'none';
      document.style.margin = '0';
    }

    // CRITICAL: Fix sidebar background and ensure it's visible
    const sidebar = document.querySelector('.wy-nav-side');
    if (sidebar) {
      sidebar.style.backgroundColor = '#343131';
      sidebar.style.color = '#9b9b9b';
      sidebar.style.width = '240px';
      sidebar.style.zIndex = '200';
      sidebar.style.position = 'fixed';
    }

    // Fix search box and sidebar content colors
    const sidebarLinks = document.querySelectorAll('.wy-nav-side .wy-menu-vertical a');
    sidebarLinks.forEach(link => {
      if (!link.style.color) {
        link.style.color = '#e2e2e2';
      }
    });
  }

  // Run immediately and on DOM ready
  fixLayout();
  document.addEventListener('DOMContentLoaded', fixLayout);

  // Also run after a short delay to ensure theme CSS is loaded
  setTimeout(fixLayout, 100);

  // Run on resize
  window.addEventListener('resize', fixLayout);
});
