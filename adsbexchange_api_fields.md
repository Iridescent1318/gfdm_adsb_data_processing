# ADS-B Exchange API Fields
## Status Information

`msg`: Shows if there is an error, default is “No error”.
`now`: The time this file was generated, in seconds since Jan 1 1970 00:00:00 GMT (the Unix epoch)
`total`: Total aircraft returned.
`ctime`: The time this file was cached, in seconds since Jan 1 1970 00:00:00 GMT (the Unix epoch).
`ptime`: The server processing time this request required in milliseconds. 

## Aircraft Information

For each aircraft object, the following are shown if available.

**`now`: the time this file was generated, in seconds since Jan 1 1970 00:00:00 GMT (the Unix epoch).**
**`messages`: the total number of Mode S messages processed (arbitrary)**
**`aircraft`: an array of JSON objects, one per known aircraft. Each aircraft has the following keys. Keys will be omitted if data is not available.**

### Detailed Aircraft Information 

**`hex`: the 24-bit ICAO identifier of the aircraft, as 6 hex digits. The identifier may start with ‘~’, this means that the address is a non-ICAO address (e.g. from TIS-B).**
`r`: aircraft registration pulled from database
`t`: aircraft type pulled from database
`dbFlags`: bitfield for certain database flags, below & must be a bitwise and … check the documentation for your programming language:

```C++
   military = dbFlags & 1;
   interesting = dbFlags & 2;
   PIA = dbFlags & 4;
   LADD = dbFlags & 8;
```
**`type`: type of underlying messages / best source of current data for this position / aircraft: ( order of which data is preferentially used )**
`adsb_icao`: messages from a Mode S or ADS-B transponder, using a 24-bit ICAO address
`adsb_icao_nt`: messages from an ADS-B equipped “non-transponder” emitter e.g. a ground vehicle, using a 24-bit ICAO address
`adsr_icao`: rebroadcast of ADS-B messages originally sent via another data link e.g. UAT, using a 24-bit ICAO address
`tisb_icao`: traffic information about a non-ADS-B target identified by a 24-bit ICAO address, e.g. a Mode S target tracked by secondary radar
`adsc`: ADS-C (received by monitoring satellite downlinks)
`mlat`: MLAT, position calculated arrival time differences using multiple receivers, outliers and varying accuracy is expected.
`other`: miscellaneous data received via Basestation / SBS format, quality / source is unknown.
`mode_s`: ModeS data from the planes transponder (no position transmitted)
`adsb_other`: messages from an ADS-B transponder using a non-ICAO address, e.g. anonymized address
`adsr_other`: rebroadcast of ADS-B messages originally sent via another data link e.g. UAT, using a non-ICAO address
`tisb_other`: traffic information about a non-ADS-B target using a non-ICAO address
`tisb_trackfile`: traffic information about a non-ADS-B target using a track/file identifier, typically from primary or Mode A/C radar
`flight`: callsign, the flight name or aircraft registration as 8 chars (2.2.8.2.6)
**`alt_baro`: the aircraft barometric altitude in feet as a number OR “ground” as a string**
`alt_geom`: geometric (GNSS / INS) altitude in feet referenced to the WGS84 ellipsoid
**`gs`: ground speed in knots**
`ias`: indicated air speed in knots
`tas`: true air speed in knots
`mach`: Mach number
`track`: true track over ground in degrees (0-359)
`track_rate`: Rate of change of track, degrees/second
`roll`: Roll, degrees, negative is left roll
`mag_heading`: Heading, degrees clockwise from magnetic north
`true_heading`: Heading, degrees clockwise from true north (usually only transmitted on ground, in the air usually derived from the magnetic heading using magnetic model WMM2020)
`baro_rate`: Rate of change of barometric altitude, feet/minute
`geom_rate`: Rate of change of geometric (GNSS / INS) altitude, feet/minute
`squawk`: Mode A code (Squawk), encoded as 4 octal digits
`emergency`: ADS-B emergency/priority status, a superset of the 7×00 squawks (2.2.3.2.7.8.1.1) (none, general, lifeguard, minfuel, nordo, unlawful, downed, reserved)
`category`: emitter category to identify particular aircraft or vehicle classes (values A0 – D7) (2.2.3.2.5.2)
`nav_qnh`: altimeter setting (QFE or QNH/QNE), hPa
`nav_altitude_mcp`: selected altitude from the Mode Control Panel / Flight Control Unit (MCP/FCU) or equivalent equipment
`nav_altitude_fms`: selected altitude from the Flight Manaagement System (FMS) (2.2.3.2.7.1.3.3)
`nav_heading`: selected heading (True or Magnetic is not defined in DO-260B, mostly Magnetic as that is the de facto standard) (2.2.3.2.7.1.3.7)
`nav_modes`: set of engaged automation modes: ‘autopilot’, ‘vnav’, ‘althold’, ‘approach’, ‘lnav’, ‘tcas’
**`lat`, `lon`: the aircraft position in decimal degrees**
`nic`: Navigation Integrity Category (2.2.3.2.7.2.6)
`rc`: Radius of Containment, meters; a measure of position integrity derived from NIC & supplementary bits. (2.2.3.2.7.2.6, Table 2-69)
`seen_pos`: how long ago (in seconds before “now”) the position was last updated
`track`: true track over ground in degrees (0-359)
`version`: ADS-B Version Number 0, 1, 2 (3-7 are reserved) (2.2.3.2.7.5)
`nic_baro`: Navigation Integrity Category for Barometric Altitude (2.2.5.1.35)
`nac_p`: Navigation Accuracy for Position (2.2.5.1.35)
`nac_v`: Navigation Accuracy for Velocity (2.2.5.1.19)
`sil`: Source Integity Level (2.2.5.1.40)
`sil_type`: interpretation of SIL: unknown, perhour, persample
`gva`: Geometric Vertical Accuracy (2.2.3.2.7.2.8)
`sda`: System Design Assurance (2.2.3.2.7.2.4.6)
`mlat`: list of fields derived from MLAT data
`tisb`: list of fields derived from TIS-B data
`messages`: total number of Mode S messages received from this aircraft
`seen`: how long ago (in seconds before “now”) a message was last received from this aircraft
`rssi`: recent average RSSI (signal power), in dbFS; this will always be negative.
`alert`: Flight status alert bit (2.2.3.2.3.2)
`spi`: Flight status special position identification bit (2.2.3.2.3.2)
`wd`, `ws`: wind direction and wind speed are calculated from ground track, true heading, true airspeed and ground speed
`oat`, `tat`: outer/static air temperature (C) and total air temperature (C) are calculated from mach number and true airspeed (typically somewhat inaccurate at lower altitudes / mach numbers below 0.5, calculation is inhibited for mach < 0.395)
`lastPosition`: {lat, lon, nic, rc, seen_pos} when the regular lat and lon are older than 60 seconds they are no longer considered valid, this will provide the last position and show the age for the last position. aircraft will only be in the aircraft json if a position has been received in the last 60 seconds or if any message has been received in the last 30 seconds.
`rr_lat`, `rr_lon`: If no ADS-B or MLAT position available, a rough estimated position for the aircraft based on the receiver’s estimated coordinates.
`acas_ra`: experimental, subject to change
`gpsOkBefore`: experimental, subject to change: aircraft lost GPS / GPS heavily degraded, it was working well before this timestamp, only displayed for 15 min after GPS is lost / degraded
Section references (2.2.xyz) refer to DO-260B.

## Supplementary Data Field

`lastPosition`: {lat, lon, nic, rc, seen_pos} when the regular lat and lon are older than 60 seconds they are no longer considered valid, this will provide the last position and show the age for the last position.
For type”adsb_icao” aka ADS-B messages, aircraft will only be in the aircraft json if a position has been received in the last 60 seconds or if any message has been received in the last 30 seconds. Positions of type “adsc” aka Sat ACARS messages have a timeout of a minimum 30 minutes.
