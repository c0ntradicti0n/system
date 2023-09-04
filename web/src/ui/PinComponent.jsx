import React, { useEffect } from 'react';

import LeaderLine from 'leader-line';


function createLine(bulbId, labelId, pinId) {
    const end = document.getElementById(bulbId);
    const start = document.getElementById(labelId);
    console.log({start, end}, {pinId, labelId})
    if (!start || !end) return
    const line = new LeaderLine(
        start, end,
        {color: 'lawngreen', size: 4, endPlugSize: 1.5, dash: {animation: true}, startSocket: 'right', endSocket: 'top'}
    );

    // Cleanup the leader line when the component unmounts
    return () => line.remove();
}

const PinComponent = ({ path, answer, addLabel,removeLabel }) => {
    const pinId = `pin-${path.replace(/\//g, '').replace('.', '')}`;
    const labelId = `pin-label-${path.replace(/\//g, '').replace('.', '')}`;
    const bulbId = `bulb-${path.replace(/\//g, '').replace('.', '')}`;

    const [clicked, setClicked] = React.useState(false);


    useEffect(() => {
        return createLine(bulbId, labelId, pinId);
    }, [labelId, bulbId, addLabel, removeLabel, clicked])

    useEffect(() => {
        addLabel({path, answer, createLine: () => createLine(bulbId, labelId, pinId)})
        return () => removeLabel({path, answer})
    }, [labelId])

    return (
        <div key={pinId} id={pinId} className="pin">
            <div className="pin-content">
                <div id={bulbId} onClick={() => setClicked(!clicked)} className="pin-bulb"></div>
            </div>
        </div>
    );
};

export { PinComponent}