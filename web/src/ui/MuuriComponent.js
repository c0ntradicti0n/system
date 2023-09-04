import React, { useEffect, useRef } from 'react';
import Muuri from 'muuri';
import '../muuri.css';

const MuuriComponent = ({ labels }) => {
  const gridRef = useRef(null);

  useEffect(() => {
    const grid = new Muuri(gridRef.current, {
  dragEnabled: true,
  layout: {
    fillGaps: true,
  },
  dragSortInterval: 50,
  dragStartPredicate: {
    distance: 8,
    delay: 0,
    handle: '.item'
  }
});

    // Clean up Muuri instance when component unmounts
    return () => {
      grid.destroy();
    };
  }, []);

  return (
    <div ref={gridRef} className="grid">
      {Object.entries(labels)
          .sort( (a, b) => a[0].localeCompare(b[0])).reverse()
          .map(([key, value]) => {
        return (
          <div
            key={key}
            id={"pin-label-" + key.replace(/\//g, "").replace(".", "")}
            className="item pin-text"
            onClick={value.createLine()}
          > {
            value.path
          }<br />
            {value.answer.answer}
          </div>
        );
      })}
    </div>
  );
};

export{ MuuriComponent}