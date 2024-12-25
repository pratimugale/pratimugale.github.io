// to be called by html elements (usually anchor).
// parameters: elementName - name of the element to be expanded
function expand(elementName) {
  const expandableElements = document.querySelectorAll('.'+elementName);
  expandableElements.forEach(element => {
    if (element.style.display === 'none' || element.style.display === '') {
      element.style.display = 'block'; 
    } else {
      element.style.display = 'none';
    }

  });
}