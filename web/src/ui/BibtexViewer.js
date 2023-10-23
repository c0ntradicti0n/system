import React from 'react';
import { toJSON } from 'bibtex-parse-js';

const BibTeXViewer = ({ entry }) => {
    let parsedData;
    console.log(entry);
    try {
        parsedData = toJSON(entry);
    } catch (e) {
        console.log(e);
        console.log(entry);
        return <div>{entry}</div>;
    }

    if (!parsedData || !parsedData.length) {
        return <div>{entry}</div>;
    }
    // Assuming only one entry for simplicity
    const data = parsedData[0].entryTags;

    return (
        <div className="bibtex-entry">
            <strong>{data.title}</strong>  {data.author}
            <em>{data.journal}</em>  {data.year}
            {data.url && <a href={data.url}> Link</a>}
        </div>
    );
}

export default BibTeXViewer;