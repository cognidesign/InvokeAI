import React, { ChangeEvent, useState } from 'react';
import { SAMPLERS } from '../../../app/constants';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAISelect from '../../../common/components/IAISelect';
import { setSampler, setSamplers } from '../optionsSlice';
import { fontSize } from './MainOptions';
import { MultiSelect } from 'react-multi-select-component';
import DualListBox from 'react-dual-listbox';
import 'react-dual-listbox/lib/react-dual-listbox.css';

// const options = [
//   { value: 'one', label: 'Option One' },
//   { value: 'two', label: 'Option Two' },
// ];

const options = SAMPLERS.map((sampler) => {
  return { value: sampler, label: sampler };
});

class Widget extends React.Component {
  state = {
    selected: [],
  };

  onChange = (selected: string[]) => {
    this.setState({ selected });
  };

  render() {
    const { selected } = this.state;

    return (
      <DualListBox
        className="main-option-black"
        options={options}
        selected={selected}
        onChange={this.onChange}
        preserveSelectOrder
        showOrderButtons
        allowDuplicates
      />
    );
  }
}

export default function MainMultiSampler() {
  const samplers = useAppSelector((state: RootState) => state.options.samplers);
  const dispatch = useAppDispatch();
  // const [selected, setSelected] = useState([]);

  // const handleChangeSamplers = (e: ChangeEvent<HTMLSelectElement>) => {
  //   // dispatch(setSampler(e.target.value));
  //   setSelected(e.target.value);
  // }
  const handleChangeSamplers = (selected: string[]) => {
    // setSelected(selected);
    dispatch(setSamplers(selected));
  };

  return (
    <div className="main-options" style={{ width: '22.5rem', color: 'white' }}>
      <DualListBox
        className="main-option-black"
        options={options}
        selected={samplers}
        onChange={handleChangeSamplers}
        preserveSelectOrder
        showOrderButtons
        allowDuplicates
      />
      {/* <MultiSelect
        labelledBy="Multi Sampler"
        options={SAMPLERS.map((sampler) => {
          return { label: sampler, value: sampler };
        })}
        value={selected}
        onChange={setSelected}
        className="main-option-block"
      /> */}
    </div>
    // <IAISelect
    //   label="Sampler"
    //   value={sampler}
    //   onChange={handleChangeSampler}
    //   validValues={SAMPLERS}
    //   fontSize={fontSize}
    //   styleClass="main-option-block"
    // />
  );
}
